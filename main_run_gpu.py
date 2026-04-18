import argparse
import logging
from pathlib import Path
import lightning as L

from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml
from SkiNet.ML.transformations.plot_transformed_data import visualize_augmented_data
from SkiNet.ML.model.lightning_model import build_lightning_model
from SkiNet.Utils.logging.logging_callbacks_setup import setup_logging_and_callbacks
from SkiNet.ML.configs.experiment_config import ExperimentConfig
from SkiNet.Utils.mlops.optuna_utils import _collect_trainer_metrics
from SkiNet.Utils.mlops.lightning_utils import configure_reproducibility
from SkiNet.ML.dataloaders.create_dataloaders import DataLoaders, create_segmentation_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_and_evaluate(main_config: ExperimentConfig, *, visualize: bool = True) -> dict[str, float]:
    """
    Run fit/test and return scalar trainer metrics.
    """
    train_cfg = main_config.trainconfig
    deterministic = configure_reproducibility(main_config)
    dataloaders: DataLoaders = create_segmentation_dataloaders(main_config)

    if visualize:
        visualize_augmented_data(dataset=dataloaders.train.dataset, samples=20)
        # restore RNG state after visualization to avoid affecting training reproducibility
        L.seed_everything(train_cfg.seed, workers=True)

    light_model = build_lightning_model(main_config)
    trainersetup = setup_logging_and_callbacks(main_config=main_config)

    # Use resolved deterministic value (bool or "warn") for backend compatibility.
    light_trainer = L.Trainer(fast_dev_run=False,  # runs 1 train + 1 val batch only
                              max_epochs=train_cfg.max_epochs,
                              logger=trainersetup.loggers,
                              callbacks=trainersetup.callbacks,
                              accelerator=train_cfg.accelerator,
                              devices=train_cfg.devices,
                              log_every_n_steps=train_cfg.log_every_n_steps,
                              deterministic=deterministic)

    try:
        light_trainer.fit(light_model, train_dataloaders=dataloaders.train, val_dataloaders=dataloaders.val)

        # Save fit metrics first because test() may mutate trainer metric stores.
        fit_metrics = _collect_trainer_metrics(light_trainer)

        if train_cfg.run_test_after_fit:
            if train_cfg.test_on_val_split:
                light_trainer.test(light_model, dataloaders=dataloaders.val)
            else:
                light_trainer.test(light_model, dataloaders=dataloaders.test)

        # Collect post-test metrics; keep fit metrics on key collisions (HPO uses val_* from fit).
        metrics = _collect_trainer_metrics(light_trainer)

        # Flush logger buffers (important for MLflow nested runs).
        for lg in light_trainer.loggers:
            if hasattr(lg, "finalize"):
                lg.finalize("success")

        return {**metrics, **fit_metrics}

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    finally:
        release_gpu()


def release_gpu() -> None:
    """Switch Lightning Studio back to CPU to stop GPU billing."""
    try:
        from lightning_sdk import Studio
        studio = Studio()
        studio.switch_machine("cpu")
        logger.info("Successfully switched to CPU machine.")
    except Exception as e:
        logger.warning(f"Could not switch machine automatically: {e}")


def main(cfg_path: Path) -> dict[str, float]:
    """
    Main function to run the pipeline from a config file.
    """
    main_config: ExperimentConfig = load_config_from_yaml(cfg_path)
    return train_and_evaluate(main_config)


if __name__ == "__main__":
    """
    Main entry point.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, help="Path to experiment YAML config")
    args = ap.parse_args()
    main(args.config)
