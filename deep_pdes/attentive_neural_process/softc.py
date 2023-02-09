from typing import Union

import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import Tensor
from torch.optim import Adam

from deep_pdes.attentive_neural_process.anp import ANP


class StefanPressureFn:
    def __init__(self, k_max: float) -> None:
        self.k_max = k_max

    def __call__(self, p: Tensor, params: Tensor) -> Tensor:
        p_stars = params[:, 0]
        return (p >= p_stars) * self.k_max


class PMEPressureFn:
    def __call__(self, p: Tensor, params: Tensor) -> Tensor:
        degrees = params[:, 0]
        return torch.relu(p).pow(degrees)


class HeatPressureFn:
    def __call__(self, p: Tensor, params: Tensor) -> Tensor:
        conductivities = params[:, 0]
        return conductivities.expand(*p.shape)


PressureFn = Union[StefanPressureFn, PMEPressureFn, HeatPressureFn]


class PINP(pl.LightningModule):
    def __init__(self, anp: ANP, pressure_fn: PressureFn, pinns_lambda: float = 1.0, lr=1e-3):
        super().__init__()
        self.anp = anp
        self.pressure_fn = pressure_fn
        self.pinns_lambda = pinns_lambda
        self.lr = lr

    def get_loc_and_scale_batched(
        self, input_contexts, output_contexts, input_targets, n_samples=100, batch_size=10000
    ):
        return self.anp.get_loc_and_scale_batched(
            input_contexts,
            output_contexts,
            input_targets,
            n_samples=n_samples,
            batch_size=batch_size,
        )

    def training_step(self, batch, batch_idx, train=True):
        context_x, context_y = batch["input_contexts"], batch["output_contexts"]
        target_x, target_y = batch["input_targets"], batch["output_targets"]
        target_x.requires_grad_(True)
        target_y_dist, _, anp_loss, _ = self.anp.forward(context_x, context_y, target_x, target_y)

        if train:
            params = rearrange(batch["params"], "nf d -> nf d 1 1")
            p = target_y_dist.loc
            pinns_loss = self._get_pinns_loss(params, p, target_x)
        else:
            pinns_loss = 0

        if train:
            self.log("train_anp_loss", anp_loss)
            self.log("train_pinn_loss", pinns_loss)
        return anp_loss + self.pinns_lambda * pinns_loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, train=False)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def _get_pinns_loss(self, params: Tensor, p: Tensor, target_x: Tensor):
        p_d = partial_deriv(p, target_x)
        p_t, p_x = p_d.split((1, 1), -1)
        k_times_p_x = self.pressure_fn(p, params) * p_x
        k_times_p_x_d = partial_deriv(k_times_p_x, target_x)
        k_times_p_x_x = k_times_p_x_d[:, :, 1:2]
        f_pred = p_t - k_times_p_x_x
        return torch.mean(f_pred**2)


def partial_deriv(out_tensor: Tensor, in_tensor: Tensor):
    return torch.autograd.grad(
        out_tensor,
        in_tensor,
        grad_outputs=torch.ones_like(out_tensor),
        retain_graph=True,
        create_graph=True,
    )[0]


class GPMEDifferentialPenalty:
    def __init__(self, pressure_fn: PressureFn) -> None:
        self.pressure_fn = pressure_fn

    def get_pinns_loss(self, params: Tensor, p: Tensor, target_x: Tensor):
        p_d = partial_deriv(p, target_x)
        p_t, p_x = p_d.split((1, 1), -1)
        k_times_p_x = self.pressure_fn(p, params) * p_x
        k_times_p_x_d = partial_deriv(k_times_p_x, target_x)
        k_times_p_x_x = k_times_p_x_d[:, :, 1:2]
        f_pred = p_t - k_times_p_x_x
        return torch.mean(f_pred**2)


class LinearAdvectionDifferentialPenalty:
    def get_pinns_loss(self, params: Tensor, p: Tensor, target_x: Tensor):
        p_d = partial_deriv(p, target_x)
        p_t, p_x = p_d.split((1, 1), -1)
        beta = params
        f_pred = p_t + beta * p_x
        return torch.mean(f_pred**2)


class BurgersDifferentialPenalty:
    def get_pinns_loss(self, params: Tensor, p: Tensor, target_x: Tensor):
        p_d = partial_deriv(p, target_x)
        p_t, p_x = p_d.split((1, 1), -1)

        # Derivative of 0.5 * p**2 wrt x
        p2_x = p * p_x
        f_pred = p_t + p2_x
        return torch.mean(f_pred**2)


DifferentialPenalty = Union[GPMEDifferentialPenalty, LinearAdvectionDifferentialPenalty]


class SoftcANP(pl.LightningModule):
    def __init__(
        self,
        anp: ANP,
        differential_penalty: DifferentialPenalty,
        pinns_lambda: float = 1.0,
        lr=1e-3,
    ):
        super().__init__()
        self.anp = anp
        self.differential_penalty = differential_penalty
        self.pinns_lambda = pinns_lambda
        self.lr = lr

    def get_loc_and_scale_batched(
        self, input_contexts, output_contexts, input_targets, n_samples=100, batch_size=10000
    ):
        return self.anp.get_loc_and_scale_batched(
            input_contexts,
            output_contexts,
            input_targets,
            n_samples=n_samples,
            batch_size=batch_size,
        )

    def training_step(self, batch, batch_idx, train=True):
        context_x, context_y = batch["input_contexts"], batch["output_contexts"]
        target_x, target_y = batch["input_targets"], batch["output_targets"]
        target_x.requires_grad_(True)
        target_y_dist, _, anp_loss, _ = self.anp.forward(context_x, context_y, target_x, target_y)

        if train:
            params = rearrange(batch["params"], "nf d -> nf d 1 1")
            p = target_y_dist.loc
            pinns_loss = self.differential_penalty.get_pinns_loss(params, p, target_x)
        else:
            pinns_loss = 0

        if train:
            self.log("train_anp_loss", anp_loss)
            self.log("train_pinn_loss", pinns_loss)
        return anp_loss + self.pinns_lambda * pinns_loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, train=False)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
