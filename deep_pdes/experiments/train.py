from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def train(cfg):  # noqa: WPS210

    dataset_path = Path(cfg.train.dataset_path)

    model = instantiate(cfg.train.model)
    train_load_path = dataset_path / "train.pt"
    valid_load_path = dataset_path / "valid.pt"
    train_dataset = instantiate(cfg.train.datasets.train, load_path=train_load_path)
    valid_dataset = instantiate(cfg.train.datasets.valid, load_path=valid_load_path)

    train_loader = train_dataset.dataloader()
    val_loader = valid_dataset.dataloader()

    checkpoint_callback: ModelCheckpoint = instantiate(cfg.train.checkpoint_callback)
    trainer: Trainer = instantiate(cfg.train.trainer, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)

    if cfg.train.save_best_model:
        model_checkpoint = torch.load(checkpoint_callback.best_model_path, map_location="cpu")

        model_state_dict = model_checkpoint["state_dict"]
    else:
        model_state_dict = model.state_dict()
    state_dict_path = Path(cfg.train.state_dict_path)
    if not state_dict_path.parent.exists():
        state_dict_path.parent.mkdir(parents=True)
    torch.save(model_state_dict, state_dict_path)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    train(cfg)


if __name__ == "__main__":
    main()
