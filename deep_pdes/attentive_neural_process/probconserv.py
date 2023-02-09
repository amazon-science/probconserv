from typing import Dict, Literal, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange, reduce
from scipy.optimize import nnls
from torch import Tensor, nn
from torch.distributions import Normal
from torch.optim import Adam
from tqdm import tqdm

from deep_pdes.attentive_neural_process.anp import ANP

LimitingMode = Literal["physnp", "hcnp"]


class PhysNP(pl.LightningModule):
    def __init__(  # noqa: WPS211
        self,
        anp: ANP,
        lr: float = 1e-4,
        constraint_precision_train=1e2,
        min_var_sample=1e-8,
        train_precision=True,
        riemann_type="trapezoid",
        non_linear_ineq_constraint=False,
        second_deriv_alpha: Optional[float] = None,
        limiting_mode: Optional[LimitingMode] = None,
        return_full_cov: bool = True,
        use_double_on_constraint: bool = True,
    ) -> None:
        super().__init__()
        self.anp = anp
        self.lr = lr
        self.train_precision = train_precision
        if self.train_precision:
            self.log_constraint_precision_train = nn.Parameter(
                torch.tensor(constraint_precision_train).log()
            )
        else:
            self.register_buffer(
                "log_constraint_precision_train", torch.tensor(constraint_precision_train).log()
            )
        self.min_var_sample = min_var_sample
        self.riemann_type = riemann_type
        self.non_linear_ineq_constraint = non_linear_ineq_constraint
        self.second_deriv_alpha = second_deriv_alpha
        self.limiting_mode = limiting_mode
        self.return_full_cov = return_full_cov
        self.use_double_on_constraint = use_double_on_constraint

    @property
    def constraint_precision_train(self):
        return self.log_constraint_precision_train.exp()

    def forward(self, context_x, context_y, target_x, mass_rhs, target_y=None):
        context_x_flat = rearrange(context_x, "b nt nx d -> b (nt nx) d")
        context_y_flat = rearrange(context_y, "b nt nx d -> b (nt nx) d")
        target_x_flat = rearrange(target_x, "b nt nx d -> b (nt nx) d")
        if target_y is not None:
            target_y_flat = rearrange(target_y, "b nt nx d -> b (nt nx) d")
        else:
            target_y_flat = None

        target_y_dist, kl, _, z = self.anp.forward(
            context_x_flat, context_y_flat, target_x_flat, target_y_flat
        )
        if target_y is not None:
            log_prob = self._constrained_log_prob(target_y_dist, target_x, target_y, mass_rhs, z)
            recon_loss = -log_prob.sum()
            loss = recon_loss + kl
        else:
            kl = None
            loss = None
        return target_y_dist, kl, loss, z

    def training_step(self, batch, batch_idx, train=True):
        out = {}
        nt = batch["n_targets_t"][0].item()
        for k in ("input_contexts", "output_contexts", "input_targets", "output_targets"):
            out[k] = rearrange(batch[k], "nf (nt nx) d -> nf nt nx d", nt=nt)
        context_x, context_y = out["input_contexts"], out["output_contexts"]
        target_x, target_y = out["input_targets"], out["output_targets"]
        mass_rhs = batch["mass_rhs"]
        _, _, loss, _ = self.forward(context_x, context_y, target_x, mass_rhs, target_y)
        if train:
            self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, train=False)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def get_loc_and_scale_batched(  # noqa: WPS210
        self,
        input_contexts,
        output_contexts,
        input_targets,
        mass_rhs,
        n_samples=100,
        batch_size=500,
    ):
        m_list = []
        s_list = []
        cov_list = []
        nf, nt, nx, _ = input_targets.shape
        n_batches = nt // batch_size + 1
        for i in tqdm(range(n_batches)):
            t_strt = i * batch_size
            t_end = (i + 1) * batch_size
            dist, cov = self.get_loc_and_scale(
                input_contexts,
                output_contexts,
                input_targets[:, t_strt:t_end],
                mass_rhs[:, t_strt:t_end],
                n_samples,
            )
            m_list.append(dist.loc.cpu())
            s_list.append(dist.scale.cpu())
            if cov is not None:
                cov_list.append(cov.cpu())
        loc = torch.cat(m_list, 1)
        scale = torch.cat(s_list, 1)
        # Check if we are returning covariance.
        if len(cov_list) > 0:
            cov = torch.cat(cov_list, 1)
        else:
            cov = None
        return Normal(loc, scale), cov

    def get_loc_and_scale(
        self, input_contexts, output_contexts, input_targets, mass_rhs, n_samples
    ):
        dist, cov = self.sample(
            input_contexts, output_contexts, input_targets, mass_rhs, n_samples=n_samples
        )
        loc = dist.loc.mean(0)
        if dist.loc.shape[0] == 1:
            var_of_locs = 0
        else:
            var_of_locs = dist.loc.var(0)
        var_total = dist.scale.pow(2).mean(0) + var_of_locs
        scale = var_total.sqrt()

        if cov is not None:
            cov_total = cov.mean(0)
        else:
            cov_total = None

        return Normal(loc, scale), cov_total

    def sample(  # noqa: WPS210
        self, input_contexts, output_contexts, input_targets, mass_rhs, n_samples
    ):
        loc_list = []
        cov_list = [] if self.return_full_cov else None
        scale_list = []
        for _ in range(n_samples):
            dist, _, _, z = self.forward(input_contexts, output_contexts, input_targets, mass_rhs)
            loc_i, cov_i = self._apply_constraint(dist, input_targets, mass_rhs, z)
            var_i = torch.diagonal(cov_i, dim1=2, dim2=3)
            var_i = var_i.unsqueeze(-1)
            var_i = var_i.clamp_min(self.min_var_sample)
            scale_i = var_i.sqrt()
            loc_list.append(loc_i)
            if cov_list is not None:
                cov_list.append(cov_i.cpu())
            scale_list.append(scale_i)
        loc = torch.stack(loc_list)
        scale = torch.stack(scale_list)
        if cov_list is not None:
            cov = torch.stack(cov_list)
        else:
            cov = None
        return Normal(loc, scale), cov

    def _apply_constraint(  # noqa: WPS210
        self, target_y_dist, target_inputs: Tensor, mass_rhs, z: Tensor
    ):
        # target_inputs: nf nt nx 2
        # target_outputs: nf nt nx 1
        # mass_rhs: nf nt
        nf, nt, nx, _ = target_inputs.shape

        mu = rearrange(target_y_dist.loc, "nf (nt nx) 1 -> nf nt nx 1", nt=nt, nx=nx)
        masses_at_t = rearrange(mass_rhs, "nf nt -> nf nt 1 1")

        input_grid = rearrange(target_inputs, "nf nt nx d -> nf nt nx d", nt=nt, nx=nx)
        x = input_grid[:, :, :, 1]

        x_delta = self._get_riemman_delta(x)

        g = rearrange(x_delta, "nf nt nx -> nf nt 1 nx")
        precis_g = self._get_constraint_precision(z)
        precis_g = rearrange(precis_g, "nf nt -> nf nt 1 1")

        eye = torch.eye(nx, device=g.device)
        eye = rearrange(eye, "nx1 nx2 -> 1 1 nx1 nx2")
        cov = target_y_dist.scale.pow(2)
        cov = rearrange(cov, "nf (nt nx) 1 -> nf nt nx 1", nt=nt)

        if self.second_deriv_alpha is not None:
            g2 = _get_second_deriv_mat(nx).to(g.device)
            g2 = rearrange(g2, "nxm2 nx -> 1 1 nxm2 nx")
            var_g2 = _get_second_derivative_var(cov, alpha=self.second_deriv_alpha).to(g.device)
            b = torch.zeros(1, 1, device=g2.device)
            mu, cov_mat = _apply_g(g2, var_g2, cov, mu, b)
        else:
            cov_mat = cov * eye

        if self.limiting_mode == "physnp":
            var_g = torch.zeros_like(precis_g)
        elif self.limiting_mode == "hcnp":
            var_g = torch.zeros_like(precis_g)
            cov_mat = eye
        else:
            var_g = 1 / precis_g

        if self.use_double_on_constraint:
            g = g.double()
            var_g = var_g.double()
            cov_mat = cov_mat.double()
            mu = mu.double()
            masses_at_t = masses_at_t.double()

        n_g = g.size(2)
        device = g.device
        dtype = g.dtype
        eye_g = torch.ones(1, 1, n_g, n_g, device=device, dtype=dtype)
        g_times_cov = g.matmul(cov_mat)
        gtr = g.transpose(3, 2)
        small_a = eye_g * var_g + (g_times_cov.matmul(gtr))
        rinv1 = torch.linalg.solve(small_a, g_times_cov)
        if self.limiting_mode == "hcnp":
            new_cov = cov * eye
        else:
            gtr_rinv1 = gtr.matmul(rinv1)
            new_cov = cov_mat.matmul(eye - gtr_rinv1)
        rinv2 = torch.linalg.solve(small_a, g.matmul(mu) - masses_at_t)
        new_mu = mu - cov_mat.matmul(gtr.matmul(rinv2))

        if self.non_linear_ineq_constraint:
            raise NotImplementedError()
        return new_mu.float(), new_cov.float()

    def _constrained_log_prob(  # noqa: WPS210
        self,
        target_y_dist: Normal,
        target_inputs: Tensor,
        target_outputs: Tensor,
        mass_rhs: Tensor,
        z: Tensor,
    ) -> Tensor:
        # target_inputs: b nt nx 2
        # target_outputs: b nt nx 1
        # mass_rhs: b nt
        b, nt, nx, _ = target_inputs.shape
        target_outputs_flat = rearrange(target_outputs, "nf nt nx 1 -> nf (nt nx) 1")
        prior_log_prob_flat = self.anp.log_prob(target_y_dist, target_outputs_flat)
        prior_log_prob = rearrange(prior_log_prob_flat, "nf (nt nx) 1 -> nf nt nx 1", nt=nt, nx=nx)

        input_grid = rearrange(target_inputs, "b nt nx d -> b nt nx d", nt=nt, nx=nx)
        output_grid = rearrange(target_outputs, "b nt nx 1 -> b nt nx", nt=nt, nx=nx)

        x = input_grid[:, :, :, 1]
        x_delta = self._get_riemman_delta(x)

        mean_constraint = (x_delta * output_grid).sum(-1)  # b nt
        precis_constraint = self._get_constraint_precision(z)
        sd_constraint = precis_constraint.pow(-0.5)
        constraint_dist = Normal(mean_constraint, sd_constraint)
        constraint_log_prob = constraint_dist.log_prob(mean_constraint)

        mu: Tensor = rearrange(target_y_dist.loc, "n (nt nx) 1 -> n nt nx", nt=nt, nx=nx)
        sd: Tensor = rearrange(target_y_dist.scale, "n (nt nx) 1 -> n nt nx", nt=nt, nx=nx)
        variance = sd.pow(2)
        mean_normalizing_constant_dist = (x_delta * mu).sum(-1)
        x_delta_squared = x_delta.pow(2)
        var_normalizing_constant_dist = (x_delta_squared * variance).sum(-1) + sd_constraint.pow(2)
        sd_normalizing_constant_dist = var_normalizing_constant_dist.sqrt()

        normalizing_constant_dist = Normal(
            mean_normalizing_constant_dist, sd_normalizing_constant_dist
        )
        normalizing_constant = normalizing_constant_dist.log_prob(mean_constraint)

        return (
            reduce(prior_log_prob, "nf nt nx 1 -> nf", "sum")
            + reduce(constraint_log_prob, "nf nt -> nf", "sum")
            - reduce(normalizing_constant, "nf nt -> nf", "sum")
        )

    def _get_riemman_delta(self, x):
        x_diff = torch.diff(x, dim=2)
        assert torch.all(x_diff >= 0)
        zero_pad_shape = (*x.shape[:2], 1)
        zero_pad = torch.zeros(*zero_pad_shape, device=x.device)
        x_delta_l: Tensor = torch.cat((x_diff, zero_pad), dim=2)
        x_delta_r: Tensor = torch.cat((zero_pad, x_diff), dim=2)
        if self.riemann_type == "trapezoid":
            x_delta = 0.5 * (x_delta_l + x_delta_r)
        elif self.riemann_type == "rhs":
            x_delta = x_delta_r
        else:
            return NotImplementedError()
        return x_delta

    def _get_constraint_precision(self, z):
        # z: nf d_z
        # max_delta: nf nt
        # precis_f: nf nt
        return self.constraint_precision_train.reshape(1, 1)


def _apply_g(g, var_g, cov, mu, mass_rhs):  # noqa: WPS210
    _, _, nx, _ = mu.shape
    _, _, ng, _ = g.shape
    eye = torch.eye(nx, device=g.device)
    eye = rearrange(eye, "nx1 nx2 -> 1 1 nx1 nx2")
    eye_g = torch.eye(ng, device=g.device)
    eye_g = rearrange(eye_g, "ng1 ng2 -> 1 1 ng1 ng2")
    gtr = g.transpose(3, 2)
    small_a = eye_g * var_g + (g.matmul(cov * gtr))
    rinv1 = torch.linalg.solve(small_a, g.matmul(cov * eye))
    new_cov = cov * (eye - gtr.matmul(rinv1))

    b = mass_rhs.unsqueeze(-1).unsqueeze(-1)
    rinv2 = torch.linalg.solve(small_a, g.matmul(mu) - b)
    new_mu = mu - cov * gtr.matmul(rinv2)
    return new_mu, new_cov


def _get_second_deriv_mat(nx):
    eye = torch.eye(nx)
    eye1 = eye[:-2]
    eye2 = eye[1:-1] * -2
    eye3 = eye[2:]
    return eye1 + eye2 + eye3


def _get_second_deriv_mat_autocor(nx, alpha=0.5):
    eye = torch.eye(nx)
    eye1 = eye[:-2] + ((alpha - 2) * alpha)
    eye2 = eye[1:-1] * -2 + alpha
    eye3 = eye[2:]
    return eye1 + eye2 + eye3


def _get_second_derivative_var(cov: Tensor, alpha=0.5):
    nf, nt, nx, _ = cov.shape
    cov0 = cov[:, :, :-2]
    cov1 = cov[:, :, 1:-1]
    cov2 = cov[:, :, 2:]

    return (
        cov0
        + 4 * cov1
        + cov2
        - 4 * alpha * cov0.sqrt() * cov1.sqrt()
        + 2 * (alpha**2) * cov0.sqrt() * cov2.sqrt()
        - 4 * alpha * cov1.sqrt() * cov2.sqrt()
    )


def get_mu_tilde_as_projection(mu, variance):
    mu = rearrange(mu, "n nt nx 1 -> n nt nx")
    variance = rearrange(variance, "n nt nx 1 -> n nt nx")
    mu_hat = reduce(mu, "n nt nx -> n nt 1", "mean")
    var_mean = reduce(variance, "n nt nx -> n nt 1", "mean")
    vec = variance / var_mean
    return mu - mu_hat * vec


def get_cov_tilde_as_projection(variance):
    variance = rearrange(variance, "n nt nx 1 -> n nt nx")
    n, nt, nx = variance.shape
    eye = torch.eye(nx, device=variance.device)
    eye = rearrange(eye, "nx1 nx2 -> 1 1 nx1 nx2")
    cov_hat = reduce(variance.unsqueeze(-1) * eye, "n nt nx1 nx2 -> n nt 1 nx2", "mean")
    var_mean = reduce(variance, "n nt nx -> n nt 1 1", "mean")
    vec = rearrange(variance, "n nt nx -> n nt nx 1") / var_mean
    return variance.unsqueeze(-1) * eye - cov_hat * vec


InequalityConstraint = Literal["monotone", "nonneg"]


def apply_non_linear_ineq_constraint(
    new_mu: Tensor, new_cov: Tensor, tol=1e-8, max_iter=1, mode: InequalityConstraint = "monotone"
):
    #  Return mean truncated to be decreasing and non-zero.
    loc = rearrange(new_mu, "nf nt nx 1 -> nf nt nx")
    nf, nt, nx = loc.shape
    new_loc_list = []
    for n in tqdm(range(nf), desc="Applying constraint"):
        loc_n = loc[n]
        new_cov_n = new_cov[n]
        new_locs_n = _apply_non_linear_constraint_one_f(loc_n, new_cov_n, max_iter, tol, mode)
        new_loc_list.append(new_locs_n)
    new_locs = torch.stack(new_loc_list, dim=0)
    return rearrange(new_locs, "nf nt nx 1 -> nf nt nx 1")


def _apply_non_linear_constraint_one_f(
    loc_n: Tensor, new_cov_n: Tensor, max_iter: int, tol: float, mode: InequalityConstraint
):
    new_locs_i = []
    nt, nx = loc_n.shape
    for t in range(nt):
        loc_t = loc_n[t].unsqueeze(-1)
        cov_t = new_cov_n[t]
        new_loc_t = _apply_non_linear_constraint_at_t(loc_t, cov_t, max_iter, tol, mode)
        new_locs_i.append(new_loc_t)
    return torch.stack(new_locs_i, dim=0)


def _apply_non_linear_constraint_at_t(  # noqa: WPS210, WPS231
    loc_t: Tensor, cov_t: Tensor, max_iter: int, tol: float, mode: InequalityConstraint
):
    nx = loc_t.shape[0]
    eye = torch.eye(nx).to(loc_t.device)
    chol_t: Tensor = torch.linalg.cholesky(cov_t)
    chinv: Tensor = torch.linalg.solve_triangular(chol_t, eye, upper=False)
    chinv_loc_t = chinv.matmul(loc_t)
    if mode == "nonneg":
        a_matrix = chinv.numpy()
        b = chinv_loc_t.squeeze(-1).numpy()
        loc_t_np, _ = nnls(a_matrix, b)
        return torch.from_numpy(loc_t_np).unsqueeze(-1)
    if mode == "monotone":
        diff_matrix = _construct_diff_matrix(nx)
        diff_matrix_inv = np.linalg.inv(diff_matrix)
        a_matrix = np.matmul(chinv.numpy(), diff_matrix_inv)
        b = chinv_loc_t.squeeze(-1).numpy()
        loc_t_np_diff, _ = nnls(a_matrix, b)
        loc_t_np = np.matmul(diff_matrix_inv, loc_t_np_diff)
        return torch.from_numpy(loc_t_np).unsqueeze(-1)


_diff_matrices: Dict[int, np.ndarray] = {}


def _construct_diff_matrix(nx: int):
    try:
        diff_matrix = _diff_matrices[nx]
    except Exception:
        eye = np.eye(nx)
        eye2 = np.eye(nx, nx, k=1)
        diff_matrix = eye - eye2
        _diff_matrices[nx] = diff_matrix
    return diff_matrix
