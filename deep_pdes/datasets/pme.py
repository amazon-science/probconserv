from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange, repeat
from scipy.optimize import root_scalar
from scipy.special import erf
from torch import Tensor  # noqa: WPS458
from torch.nn import functional

from deep_pdes.datasets.base import (
    ANPDataset,
    meshgrid,
    sample_points_from_each_interval,
    sample_points_uniformly,
)
from deep_pdes.datasets.pbc import convection_diffusion_solution


class GeneralizedPorousMediumEquation(ANPDataset):  # noqa: WPS214
    def __init__(  # noqa: WPS210, WPS211
        self,
        n_functions: int,
        n_contexts_t: int,
        n_contexts_x: int,
        n_targets_t: int,
        n_targets_x: int,
        batch_size: int,
        t_range: Tuple[float, float] = (0, 1),
        x_range: Tuple[float, float] = (0, 1),
        load_path: Optional[str] = None,
    ):
        self._batch_size = batch_size
        self.n_contexts_t = n_contexts_t
        self.n_contexts_x = n_contexts_x
        self.n_targets_t = n_targets_t
        self.n_targets_x = n_targets_x
        self.t_range = t_range
        self.x_range = x_range
        if load_path is not None:
            tensors, parameters = torch.load(load_path)  # noqa: WPS110
        else:
            tensors = None
            parameters = self._sample_parameters(n_functions)  # noqa: WPS110
        self._parameters = parameters  # noqa: WPS110
        if tensors is None:
            tensors = self._make_solution(
                n_functions,
                n_contexts_t,
                n_contexts_t,
                n_targets_t,
                n_targets_x,
            )
        self._tensors: Dict[str, Tensor] = tensors
        self.mass_rhs = self._mass_rhs()

    @property
    def dimnames(self):
        return ("t", "x")

    def lims(self, dimname: str):
        if dimname == "x":
            return self.x_range
        elif dimname == "t":
            return self.t_range
        raise ValueError()

    @property
    def tensors(self):
        return self._tensors

    @property
    def parameters(self):  # noqa: WPS110
        return self._parameters

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def has_mass_rhs(self) -> bool:
        return True

    @abstractmethod
    def solution(self, inputs: Tensor):
        # inputs: nf (nt nx) 2
        # outputs: nf (nt nx) 1
        pass  # noqa: WPS420

    def _make_solution(
        self,
        n_functions: int,
        n_contexts_t: int,
        n_contexts_x: int,
        n_targets_t: int,
        n_targets_x: int,
    ) -> Dict[str, Tensor]:
        input_sample_settings = (
            ("contexts", n_contexts_t, n_contexts_x),
            ("targets", n_targets_t, n_targets_x),
        )
        tensors: Dict[str, Tensor] = {}
        for mode, n_t, n_x in input_sample_settings:
            ts = sample_ts(n_functions, n_t, mode=mode, t_range=self.t_range)
            xs = sample_xs(n_functions, n_x, x_range=self.x_range)
            inputs = meshgrid(ts, xs)
            inputs = rearrange(inputs, "nf nt nx d -> nf (nt nx) d")
            outputs = self.solution(inputs)
            tensors.update(
                {
                    f"input_{mode}": inputs,
                    f"output_{mode}": outputs,
                }
            )
        return tensors

    @abstractmethod
    def _mass_rhs(self):
        pass  # noqa: WPS420


class PorousMediumEquation(GeneralizedPorousMediumEquation):
    def __init__(
        self,
        n_functions: int,
        *args,
        scale_lims=(0.2, 5),
        degree_min=2,
        degree_max=6,
        degrees=None,
        **kwargs,
    ):
        self.scale_lims = scale_lims
        if degrees is None:
            self._degrees = None
            self.degree_min = degree_min
            self.degree_max = degree_max
        else:
            self._degrees = torch.tensor(degrees)
            self.degree_min = self._degrees.min().item()
            self.degree_max = self._degrees.max().item()

        super().__init__(n_functions, *args, **kwargs)

    @property
    def degrees(self):
        return self._parameters[:, 0]

    @property
    def scales(self):
        return self._parameters[:, 1]

    @property
    def params(self) -> Tensor:
        return self._parameters

    def solution(self, inputs: Tensor):
        return self.true_solution(inputs, self.degrees, self.scales)

    def true_solution(self, inputs: Tensor, degrees: Tensor, scales: Tensor):
        # inputs: nf (nx nt) 2
        degrees = rearrange(degrees, "nf -> nf 1")
        scales = rearrange(scales, "nf -> nf 1")
        ts, xs = torch.split(inputs, (1, 1), -1)
        ts = ts.squeeze(-1)
        xs = xs.squeeze(-1) * scales

        us = degrees * functional.relu(ts - xs)
        ys = us.pow(1 / degrees)
        return rearrange(ys, "nf nt_nx -> nf nt_nx 1")

    def shock_points(self, i: int, x_of_interest):
        scale = self.scales[i]
        return torch.tensor(x_of_interest) * scale

    def _mass_rhs(self):
        degree = self.degrees
        degree = rearrange(degree, "nf -> nf 1")
        input_targets = rearrange(
            self.tensors["input_targets"], "nf (nt nx) d -> nf nt nx d", nt=self.n_targets_t
        )
        ts = input_targets[:, :, 0, 0]
        return mass_pme(degree, ts)

    def _sample_parameters(self, n_functions: int):
        if self._degrees is None:
            degrees = self.degree_min + torch.rand(n_functions) * (
                self.degree_max - self.degree_min
            )
        else:
            n_functions_per_degree = n_functions // self._degrees.shape[0]
            degrees = repeat(self._degrees, "nd -> (nd nfpd)", nfpd=n_functions_per_degree)
        scales = self._sample_scales(n_functions, self.scale_lims)
        return torch.stack((degrees, scales), -1)

    def _sample_scales(self, n_functions, scale_lims):
        min_scale_log = torch.tensor(scale_lims[0]).log()
        max_scale_log = torch.tensor(scale_lims[1]).log()
        scales_log = min_scale_log + torch.rand(n_functions) * (max_scale_log - min_scale_log)
        return torch.exp(scales_log)


def mass_pme(degree: Tensor, ts: Tensor):
    a1 = 1 + (1 / degree)
    return (degree.pow(a1)) / (degree + 1) * ts.pow(a1)


class Stefan:
    def __init__(self, p_star=0.5):
        self.p_star = p_star
        self.k_min = 0
        self.k_max = 1

        self._z1: Optional[float] = None
        self._alpha: Optional[float] = None

    def true_solution(self, inputs: np.ndarray):
        ts, xs = np.split(inputs, 2, -1)
        p1 = self.p1(xs, ts)
        p2 = self.p2(xs, ts)
        x_star = self.alpha * np.sqrt(ts)
        p = p1 * (xs <= x_star) + p2 * (xs > x_star)
        p[np.isclose(xs, 0)] = 1.0
        return p.squeeze(-1)

    def shock_points(self, x_of_interest):
        return np.power(x_of_interest / self.alpha, 2)

    def mass_stefan(self, ts: Tensor) -> Tensor:
        k_max = self.k_max
        c1 = self.c1
        a1: float = 2 * np.sqrt(k_max / np.pi)

        return a1 * c1 * torch.sqrt(ts)

    def p1(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        a = x / (2 * np.sqrt(self.k_max * t))
        return 1 - self.c1 * erf(a)

    def p2(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        if self.k_min == 0:
            return np.zeros_like(x)
        a = x / (2 * np.sqrt(self.k_min * t))
        return self.c2 * (1 - erf(a))

    @property
    def c1(self) -> float:
        num = 1 - self.p_star
        dem = erf(self.alpha / (2 * (np.sqrt(self.k_max))))
        return num / dem

    @property
    def c2(self) -> float:
        num = self.p_star
        a = self.alpha / (2 * np.sqrt(self.k_min))
        dem = 1 - erf(a)
        return num / dem

    @property
    def alpha(self) -> float:
        if self._alpha is None:
            self._alpha = 2 * np.sqrt(self.k_max) * self.z1
        return self._alpha

    @property
    def z1(self) -> float:
        if self._z1 is None:
            self._z1 = root_scalar(self._z1_objective, bracket=(0, 10)).root
        return self._z1

    def _z1_objective(self, z1):
        a1 = self.p_star * erf(z1)
        a2 = z1 * np.exp(np.power(z1, 2))
        b = (1 - self.p_star) / np.sqrt(np.pi)
        return (a1 * a2) - b


class StefanPME(GeneralizedPorousMediumEquation):
    def __init__(
        self,
        n_functions,
        *args,
        p_star_lim: Tuple[float, float] = (0.1, 0.9),
        p_stars: Optional[Tuple[float, ...]] = None,
        **kwargs,
    ):
        if p_stars is None:
            self._p_stars = None
            self.p_star_lim = p_star_lim
        else:
            self._p_stars = torch.tensor(p_stars)
            self.p_star_lim = (min(p_stars), max(p_stars))
        super().__init__(n_functions, *args, **kwargs)

    @property
    def stefans(self) -> List[Stefan]:
        return self._parameters

    @property
    def params(self) -> Tensor:
        params = torch.tensor([s.p_star for s in self.stefans])
        return params.reshape(-1, 1)

    def solution(self, inputs: Tensor):
        # inputs: nf (nt nx) 2
        # outputs: nf (nt nx) 1
        assert len(inputs.shape) == 3
        nf = len(self.stefans)
        if inputs.shape[0] == 1:
            inputs = inputs.expand(nf, -1, -1)
        soln_list = []
        for in_tensor, stefan in zip(inputs, self.stefans):
            soln_i = torch.from_numpy(stefan.true_solution(in_tensor.numpy()))
            soln_i = soln_i.float()
            soln_list.append(soln_i)
        return torch.stack(soln_list, dim=0).unsqueeze(-1)

    def shock_points(self, i: int, x_of_interest):
        stefan = self.stefans[i]
        shock_points = stefan.shock_points(np.array(x_of_interest))
        return torch.from_numpy(shock_points)

    def _mass_rhs(self):
        input_targets = rearrange(
            self.tensors["input_targets"], "nf (nt nx) d -> nf nt nx d", nt=self.n_targets_t
        )
        ts = input_targets[:, :, 0, 0]
        stefans = self.stefans
        masses: List[Tensor] = []
        for ts_i, stefan in zip(ts, stefans):
            mass = stefan.mass_stefan(ts_i)
            masses.append(mass)
        return torch.stack(masses)

    def _sample_parameters(self, n_functions: int):
        if self._p_stars is None:
            a = self.p_star_lim[0]
            b_minus_a = self.p_star_lim[1] - self.p_star_lim[0]
            p_stars = a + torch.rand(n_functions) * b_minus_a
        else:
            n_functions_per_pstar = n_functions // self._p_stars.shape[0]
            p_stars = repeat(self._p_stars, "npstar -> (npstar nfpps)", nfpps=n_functions_per_pstar)
        return [Stefan(p_star=p.item()) for p in p_stars]


def sample_ts(n_functions: int, n_t: int, mode: str, t_range: Tuple[float, float]) -> Tensor:
    if mode == "contexts":
        us = sample_points_from_each_interval(n_functions, n_t)
    elif mode == "targets":
        us = sample_points_uniformly(n_functions, n_t)
        us, _ = torch.sort(us)
    t_min, t_max = t_range
    return t_min + us * (t_max - t_min)


def make_dense_grid(nt, nx):
    dt = torch.linspace(0, 1, nt).unsqueeze(0)
    dx = torch.linspace(0, 1, nx).unsqueeze(0)
    return meshgrid(dt, dx)


def sample_xs(n_functions: int, n_x: int, x_range: Tuple[float, float]) -> Tensor:
    us = sample_points_uniformly(n_functions, n_x)
    us, _ = torch.sort(us)
    x_min, x_max = x_range
    return x_min + us * (x_max - x_min)


class HeatEquation(GeneralizedPorousMediumEquation):
    def __init__(
        self,
        n_functions: int,
        *args,
        conductivity_min=1,
        conductivity_max=5,
        conductivities=None,
        **kwargs,
    ):
        if conductivities is None:
            self._conductivities = None
            self.conductivity_min = conductivity_min
            self.conductivity_max = conductivity_max
        else:
            self._conductivities = torch.tensor(conductivities)
            self.conductivity_min = self._conductivities.min().item()
            self.conductivity_max = self._conductivities.max().item()

        self.nx_soln = 512
        x_range = (0, 2 * np.pi)
        kwargs["x_range"] = x_range

        super().__init__(n_functions, *args, **kwargs)

    @property
    def conductivities(self) -> Tensor:
        return self._parameters[:, 0]

    @property
    def thetas(self) -> Tensor:
        return self._parameters[:, 1]

    @property
    def params(self) -> Tensor:
        return self._parameters

    def solution(self, inputs: Tensor):
        nf = len(self.thetas)
        if inputs.shape[0] == 1:
            inputs = inputs.expand(nf, -1, -1)
        assert inputs.shape[0] == nf
        return self.true_solution(inputs, self.thetas, self.conductivities)

    def true_solution(self, inputs: Tensor, thetas: Tensor, nus: Tensor) -> Tensor:
        ts = inputs[:, :, 0]
        xs = inputs[:, :, 1]
        nf = len(thetas)
        xs = xs.unique(dim=1).reshape(nf, -1)

        ts = ts.unique(dim=1).reshape(nf, -1)
        tr_all = self.convection_onedim(ts, thetas, nus)

        nt_nx = inputs.shape[1]
        nt = int(np.sqrt(nt_nx))

        grid = rearrange(inputs, "nf (nt nx) d -> nf nt nx d", nt=nt).clone()
        grid[:, :, :, 1] /= np.pi * 2
        grid = (grid - 0.5) * 2
        # (h w) to (x y)
        grid_x = grid[:, :, :, 1]
        grid_y = grid[:, :, :, 0]
        grid = torch.stack((grid_x, grid_y), dim=-1)

        tr = functional.grid_sample(
            tr_all.unsqueeze(1).float(), grid, align_corners=True, mode="bilinear"
        ).squeeze(1)
        tr = rearrange(tr, "nf nt nx -> nf (nt nx) 1")
        return tr.float()

    def convection_onedim(self, t_values: Tensor, thetas: Tensor, nus: Tensor):
        n_function_draws = thetas.shape[0]
        u_list = []
        for i in range(n_function_draws):
            u_i = self._convection_onedim_for_one_parameter(
                t_values[i, :],
                thetas[i].item(),
                nus[i].item(),
            )
            u_list.append(u_i)
        u = np.stack(u_list, axis=0)
        return torch.from_numpy(u)

    def _convection_onedim_for_one_parameter(self, t_values, theta: float, nu: float):
        two_pi = 2 * np.pi
        dx = two_pi / self.nx_soln
        x_grid = np.arange(0, two_pi, dx)
        x_start = np.sin(x_grid + theta)
        t_grid = repeat(t_values, "nt -> nt nx", nx=self.nx_soln)
        t_grid = t_grid.numpy()
        return convection_diffusion_solution(x_start, t_grid, nu, beta=0)

    def _mass_rhs(self):
        cs = self.conductivities
        cs = rearrange(cs, "nf -> nf 1")
        return torch.zeros_like(cs)

    def _sample_parameters(self, n_functions: int):
        if self._conductivities is None:
            degrees = self.conductivity_min + torch.rand(n_functions) * (
                self.conductivity_max - self.conductivity_min
            )
        else:
            n_functions_per_degree = n_functions // self._conductivities.shape[0]
            degrees = repeat(self._conductivities, "nd -> (nd nfpd)", nfpd=n_functions_per_degree)
        thetas = torch.zeros(n_functions)
        return torch.stack((degrees, thetas), -1)

    def _weighted_average(self, xs, full_output: Tensor):
        _, nt, _ = full_output.shape
        indx_floating = xs / (2 * np.pi) * (self.nx_soln - 1)

        indx_lower = torch.floor(indx_floating).to(torch.int64)
        indx_lower = repeat(indx_lower, "nf nx -> nf nt nx", nt=nt)

        indx_higher = indx_lower + 1
        w_lower = indx_higher - indx_floating.unsqueeze(1)
        w_higher = indx_floating.unsqueeze(1) - indx_lower
        padding = torch.zeros_like(full_output)[:, :, 0:1]
        full_output = torch.cat((full_output, padding), -1)
        full_output_lower = torch.gather(full_output, dim=-1, index=indx_lower)
        full_output_higher = torch.gather(full_output, dim=-1, index=indx_higher)

        return full_output_lower * w_lower + full_output_higher * w_higher


class LinearAdvection(GeneralizedPorousMediumEquation):
    def __init__(
        self,
        n_functions,
        *args,
        a_lim: Tuple[float, float] = (1, 10),
        a_vals: Optional[Tuple[float, ...]] = None,
        **kwargs,
    ):
        if a_vals is None:
            self._a_vals = None
            self.a_lim = a_lim
        else:
            self._a_vals = torch.tensor(a_vals)
            self.a_lim = (min(a_vals), max(a_vals))
        super().__init__(n_functions, *args, **kwargs)

    @property
    def params(self) -> Tensor:
        return self._parameters.unsqueeze(-1)

    def solution(self, inputs: Tensor):
        # inputs: nf (nt nx) 2
        # outputs: nf (nt nx) 1
        assert len(inputs.shape) == 3
        nf = self.parameters.shape[0]
        if inputs.shape[0] == 1:
            inputs = inputs.expand(nf, -1, -1)
        t = inputs[:, :, 0]
        x = inputs[:, :, 1]
        a = rearrange(self.parameters, "nf -> nf 1")
        u = self.h(x - t * a)
        return rearrange(u, "nf nt_nx -> nf nt_nx 1")

    def h(self, x: Tensor):
        return (x <= 0.5).float()  # noqa: WPS459

    def mass_advection(self, ts: Tensor, a_vals: Tensor):
        max_density_tnsr = torch.tensor(0.5, device=ts.device)
        return 0.5 + torch.minimum(ts * a_vals, max_density_tnsr)

    def _mass_rhs(self):
        input_targets = rearrange(
            self.tensors["input_targets"], "nf (nt nx) d -> nf nt nx d", nt=self.n_targets_t
        )
        ts = input_targets[:, :, 0, 0]
        a_vals = self.parameters.reshape(-1, 1)
        return self.mass_advection(ts, a_vals)

    def _sample_parameters(self, n_functions: int):
        if self._a_vals is None:
            a = self.a_lim[0]
            b_minus_a = self.a_lim[1] - self.a_lim[0]
            a_vals = a + torch.rand(n_functions) * b_minus_a
        else:
            n_functions_per_pstar = n_functions // self._a_vals.shape[0]
            a_vals = repeat(self._a_vals, "npstar -> (npstar nfpps)", nfpps=n_functions_per_pstar)
        return a_vals


class Burgers(GeneralizedPorousMediumEquation):
    def __init__(
        self,
        n_functions,
        *args,
        a_lim: Tuple[float, float] = (1, 5),
        a_vals: Optional[Tuple[float, ...]] = None,
        **kwargs,
    ):
        if a_vals is None:
            self._a_vals = None
            self.a_lim = a_lim
        else:
            self._a_vals = torch.tensor(a_vals)
            self.a_lim = (min(a_vals), max(a_vals))

        x_range = (-1, 1)
        kwargs["x_range"] = x_range
        super().__init__(n_functions, *args, **kwargs)

    @property
    def params(self) -> Tensor:
        return self._parameters.unsqueeze(-1)

    def solution(self, inputs: Tensor):
        # inputs: nf (nt nx) 2
        # outputs: nf (nt nx) 1
        nf = self.parameters.shape[0]
        if inputs.shape[0] == 1:
            inputs = inputs.expand(nf, -1, -1)
        t = inputs[:, :, 0]
        x = inputs[:, :, 1]
        a = rearrange(self.parameters, "nf -> nf 1")
        u = solution_burgers(t, x, a)
        return rearrange(u, "nf nt_nx -> nf nt_nx 1")

    def _mass_rhs(self):
        input_targets = rearrange(
            self.tensors["input_targets"], "nf (nt nx) d -> nf nt nx d", nt=self.n_targets_t
        )
        ts = input_targets[:, :, 0, 0]
        a_vals = self.parameters.reshape(-1, 1)
        return mass_burgers(ts, a_vals)

    def _sample_parameters(self, n_functions: int):
        if self._a_vals is None:
            a = self.a_lim[0]
            b_minus_a = self.a_lim[1] - self.a_lim[0]
            a_vals = a + torch.rand(n_functions) * b_minus_a
        else:
            n_functions_per_pstar = n_functions // self._a_vals.shape[0]
            a_vals = repeat(self._a_vals, "npstar -> (npstar nfpps)", nfpps=n_functions_per_pstar)
        return a_vals


def solution_burgers(t: Tensor, x: Tensor, a: Tensor):
    break_time = a.pow(-1)
    u_prebreak = _solution_burgers_prebreak(t, x, a)
    u_postbreak = _solution_burgers_postbreak(t, x, a)
    return u_prebreak * (t <= break_time) + u_postbreak * (t > break_time)


def _solution_burgers_prebreak(t: Tensor, x: Tensor, a: Tensor) -> Tensor:
    c1 = x <= ((a * t) - 1)
    u1 = a * c1
    c2 = torch.logical_and(~c1, x <= 0)
    u2 = (a * x) / (a * t - 1) * c2
    # zero if above u3
    return u1 + u2


def _solution_burgers_postbreak(t: Tensor, x: Tensor, a: Tensor) -> Tensor:
    c = x <= (0.5 * (a * t - 1))
    return c * a


def mass_burgers(ts: Tensor, a_vals: Tensor):
    return (a_vals / 2) * (1 + (a_vals * ts))


if __name__ == "__main__":
    pme = PorousMediumEquation(10, 10, 10, 10, 10, 10)
