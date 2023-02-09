import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.distributions import Normal
from torch.optim import Adam
from tqdm import tqdm

from deep_pdes.attentive_neural_process.module import Decoder, DeterministicEncoder, LatentEncoder


class ANP(pl.LightningModule):  # noqa: WPS214
    def __init__(self, num_hidden, dim_x=1, dim_y=1, lr=1e-4, free_bits=None, checkpoint=None):
        super().__init__()
        self.latent_encoder = LatentEncoder(num_hidden, num_hidden, dim_x=dim_x, dim_y=dim_y)
        self.deterministic_encoder = DeterministicEncoder(
            num_hidden, num_hidden, dim_x=dim_x, dim_y=dim_y
        )
        self.decoder = Decoder(num_hidden, dim_x=dim_x, dim_y=dim_y)
        self.lr = lr
        self.free_bits = free_bits

        if checkpoint is not None:
            ckpt = torch.load(checkpoint, map_location=self.device)
            self.load_state_dict(ckpt["state_dict"])

    def forward(self, context_x, context_y, target_x, target_y=None):  # noqa: WPS210
        num_targets = target_x.size(1)

        context_mu, context_var, context_z = self.latent_encoder(context_x, context_y)

        training = target_y is not None
        if training:
            target_mu, target_var, target_z = self.latent_encoder(target_x, target_y)
            z = target_z

        # For Generation
        else:
            z = context_z

        z = z.unsqueeze(1).repeat(1, num_targets, 1)
        # sizes are [B, T_target, H]
        r = self.deterministic_encoder(context_x, context_y, target_x)
        # mu should be the prediction of target y
        target_y_dist: Normal = self.decoder(r, z, target_x)

        if training:
            # get log probability
            recon_prob = self.log_prob(target_y_dist, target_y)
            recon_loss = -recon_prob.sum()
            # get KL divergence between prior and posterior
            kl = self.kl_div(context_mu, context_var, target_mu, target_var)
            if self.free_bits is not None:
                kl = torch.clamp_min(kl, self.free_bits)

            # maximize prob and minimize KL divergence
            loss = recon_loss + kl

        # For Generation
        else:
            kl = None
            loss = None

        return target_y_dist, kl, loss, z[:, 0, :]

    def log_prob(self, target_y_dist: Normal, target_y: Tensor) -> Tensor:
        return target_y_dist.log_prob(target_y)

    def sample(self, context_x, context_y, target_x, n_samples=1):
        loc_list = []
        scale_list = []
        for _ in range(n_samples):
            dist, _, _, _ = self.forward(context_x, context_y, target_x)
            loc_list.append(dist.loc)
            scale_list.append(dist.scale)
        loc = torch.stack(loc_list)
        scale = torch.stack(scale_list)
        return Normal(loc, scale)

    def get_loc_and_scale_batched(
        self, input_contexts, output_contexts, input_targets, n_samples=100, batch_size=10000
    ):
        m_list = []
        s_list = []
        n_targets = input_targets.shape[1]
        n_batches = n_targets // batch_size + 1
        for i in tqdm(range(n_batches)):
            b_strt = i * batch_size
            b_end = (i + 1) * batch_size
            dist = self.get_loc_and_scale(
                input_contexts, output_contexts, input_targets[:, b_strt:b_end], n_samples
            )
            m_list.append(dist.loc.cpu())
            s_list.append(dist.scale.cpu())
        loc = torch.cat(m_list, 1)
        scale = torch.cat(s_list, 1)
        return Normal(loc, scale)

    def get_loc_and_scale(self, input_contexts, output_contexts, input_targets, n_samples=100):
        dist = self.sample(input_contexts, output_contexts, input_targets, n_samples=n_samples)
        loc = dist.loc.mean(0)
        if loc.shape[0] == 1:
            var_of_locs = 0
        else:
            var_of_locs = dist.loc.var(0)
        var_total = dist.scale.pow(2).mean(0) + var_of_locs
        scale = var_total.sqrt()
        return Normal(loc, scale)

    def training_step(self, batch, batch_idx, train=True):
        context_x, context_y = batch["input_contexts"], batch["output_contexts"]
        target_x, target_y = batch["input_targets"], batch["output_targets"]
        _, kl, loss, _ = self.forward(context_x, context_y, target_x, target_y)
        if train:
            self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, train=False)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = torch.exp(posterior_var) + (posterior_mu - prior_mu) ** 2
        kl_div /= torch.exp(prior_var)
        kl_div -= 1.0
        kl_div += prior_var - posterior_var
        return 0.5 * kl_div.sum()

    def grad_of_mean_wrt_target(self, context_x, context_y, target_x):
        target_y_dist, _, _ = self.forward(context_x, context_y, target_x)
        return self._grad_of_mean_wrt_target(target_y_dist, target_x).loc

    def _grad_of_mean_wrt_target(self, target_y_dist: Normal, target_x):
        mean_target_y = target_y_dist.loc
        scale_target_y = target_y_dist.scale
        grad_mean = torch.autograd.grad(
            mean_target_y,
            target_x,
            torch.ones_like(mean_target_y),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_scale = torch.autograd.grad(
            scale_target_y,
            target_x,
            torch.ones_like(mean_target_y),
            create_graph=True,
            retain_graph=True,
        )[0].abs()
        return Normal(grad_mean, grad_scale)
