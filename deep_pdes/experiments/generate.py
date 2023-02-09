from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate

from deep_pdes.datasets.base import ANPDataset


def generate(cfg):
    dataset_path = Path(cfg.datasets.save_path)
    overwrite: bool = cfg.datasets.dataset_overwrite
    dataset_path.mkdir(parents=True, exist_ok=True)
    for dataset_type in ("train", "valid", "test", "pinn_grid_train", "pinn_grid_valid"):
        path = dataset_path / f"{dataset_type}.pt"
        if not overwrite:
            assert not path.exists()
        dataset_cfg = cfg.datasets.get(dataset_type)
        if dataset_cfg is not None:
            dataset: ANPDataset = instantiate(dataset_cfg)
            data_to_save = (dataset.tensors, dataset.parameters)
            torch.save(data_to_save, path)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    generate(cfg)


if __name__ == "__main__":
    main()
