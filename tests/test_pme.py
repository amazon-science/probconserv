from hydra import compose, initialize

from deep_pdes.experiments.generate import generate
from deep_pdes.experiments.train import train


def run_pme(experiment: str, model: str, extra_overrides=None):
    with initialize(
        version_base=None, config_path="../deep_pdes/experiments/conf", job_name="test_app"
    ):

        overrides = [
            f"+experiments={experiment}",
            f"+train={experiment}_{model}",
            "base_dir=./output/test",
            "train.trainer.accelerator=",
            "train.trainer.max_epochs=2",
            "train.trainer.check_val_every_n_epoch=2",
            "datasets.dataset_overwrite=True",
            "datasets.train.n_functions=2",
            "datasets.valid.n_functions=1",
            "analysis.gpu=cpu",
            "analysis.nx=11",
            "analysis.nt=11",
        ]
        if extra_overrides is not None:
            overrides += extra_overrides
        cfg = compose(config_name="config", overrides=overrides)
        generate(cfg)
        train(cfg)


def test_anp_pme():
    run_pme("1b_pme_var_m", "anp")


def test_anp_stefan():
    run_pme("2b_stefan_var_p", "anp")


def test_physnp_pme():
    run_pme("1b_pme_var_m", "physnp")


def test_physnp_stefan():
    run_pme("2b_stefan_var_p", "physnp")
