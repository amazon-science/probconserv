from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

import icontract
import torch
from einops import repeat
from torch import Tensor  # noqa: WPS458
from torch.utils.data import DataLoader, Dataset


@icontract.invariant(lambda self: self.validate())
class ANPDataset(Dataset, icontract.DBC):  # noqa: WPS214
    def __init__(self) -> None:
        super().__init__()
        self.mass_rhs: Optional[Tensor] = None
        self.n_targets_t: Optional[int] = None
        self.n_targets_x: Optional[int] = None

    @property
    @abstractmethod
    def tensors(self) -> Dict[str, Tensor]:
        pass  # noqa: WPS420

    @property  # type: ignore
    @icontract.ensure(lambda result: (result is None) or (len(result.shape) == 2))
    def params(self) -> Optional[Tensor]:
        return None  # noqa: WPS324

    @abstractmethod
    @icontract.require(lambda inputs: len(inputs.shape) == 3)
    @icontract.ensure(lambda result: len(result.shape) == 3)
    def solution(self, inputs: Tensor) -> Tensor:
        pass  # noqa: WPS420

    @abstractmethod
    def lims(self, dimname: str) -> Tuple[float, float]:
        pass  # noqa: WPS420

    @property
    @abstractmethod
    def dimnames(self) -> Tuple[str]:
        pass  # noqa: WPS420

    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass  # noqa: WPS420

    @property
    def dimensions(self) -> Dict[str, int]:
        return {
            "n_functions": self.tensors["input_contexts"].shape[0],
            "n_contexts": self.tensors["input_contexts"].shape[1],
            "n_targets": self.tensors["input_targets"].shape[1],
            "input_dim": self.tensors["input_targets"].shape[2],
            "output_dim": self.tensors["output_contexts"].shape[2],
        }

    def validate(self):
        dim = self.dimensions
        shapes = {
            "input_contexts": (dim["n_functions"], dim["n_contexts"], dim["input_dim"]),
            "output_contexts": (dim["n_functions"], dim["n_contexts"], dim["output_dim"]),
            "input_targets": (dim["n_functions"], dim["n_targets"], dim["input_dim"]),
            "output_targets": (dim["n_functions"], dim["n_targets"], dim["output_dim"]),
        }
        for nm, shape in shapes.items():
            assert self.tensors[nm].shape == shape, f"{nm} has incorrect shape"
        return True

    def __getitem__(self, idx) -> Dict[str, Tensor]:
        tensors = {k: v[idx] for k, v in self.tensors.items()}
        if self.params is not None:
            tensors["params"] = self.params[idx]
        if self.mass_rhs is not None:
            tensors["mass_rhs"] = self.mass_rhs[idx]
            tensors["n_targets_t"] = torch.tensor(self.n_targets_t)
            tensors["n_targets_x"] = torch.tensor(self.n_targets_x)
        return tensors

    def __len__(self):
        return self.dimensions["n_functions"]

    def dataloader(self):
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=True,
        )


def sample_points_from_each_interval(n_functions: int, n_pts: int) -> Tensor:
    interval_starts = torch.tensor(range(n_pts)) / n_pts
    ts = torch.rand((n_functions, n_pts)) * (1 / n_pts)
    ts += interval_starts
    return ts


def sample_points_uniformly(n_functions: int, n_pts: int, min_val=0, max_val=1) -> Tensor:
    return sample_uniform((n_functions, n_pts), min_val, max_val)


def sample_uniform(
    size: Tuple[int, ...],
    min_val: float = 0,
    max_val: float = 1,
    sort: bool = False,
) -> Tensor:
    us = torch.rand(size)
    if sort:
        us, _ = torch.sort(us)
    return min_val + (max_val - min_val) * us


def meshgrid(ts: Tensor, xs: Tensor):
    nf, nt = ts.shape
    _, nx = xs.shape
    ts = repeat(ts, "nf nt -> nf nt nx", nx=nx)
    xs = repeat(xs, "nf nx -> nf nt nx", nt=nt)
    return torch.stack((ts, xs), dim=-1)
