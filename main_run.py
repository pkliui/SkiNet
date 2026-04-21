import argparse
import logging
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml
from SkiNet.ML.transformations.plot_transformed_data import visualize_augmented_data
from SkiNet.ML.model.lightning_model import build_lightning_model
from SkiNet.Utils.logging.logging_callbacks_setup import TrainerComponents, setup_logging_and_callbacks
from SkiNet.ML.configs.experiment_config import ExperimentConfig
from SkiNet.Utils.mlops.optuna_utils import _collect_trainer_metrics
from SkiNet.Utils.mlops.lightning_utils import configure_reproducibility
from SkiNet.ML.dataloaders.create_dataloaders import DataLoaders, create_segmentation_dataloaders
from SkiNet.ML.configs.train_configs.train_config import TrainConfig

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
                              precision=train_cfg.precision,
                              log_every_n_steps=train_cfg.log_every_n_steps,
                              deterministic=deterministic,
                              check_val_every_n_epoch=train_cfg.check_val_every_n_epoch,
                              num_sanity_val_steps=train_cfg.num_sanity_val_steps)

    light_trainer.fit(light_model, train_dataloaders=dataloaders.train, val_dataloaders=dataloaders.val)

    # Save fit metrics first because test() may mutate trainer metric stores.
    fit_metrics = _collect_trainer_metrics(light_trainer)

    if train_cfg.run_test_after_fit:
        _run_post_fit_test(light_trainer=light_trainer,
                           light_model=light_model,
                           dataloaders=dataloaders,
                           train_cfg=train_cfg,
                           trainersetup=trainersetup)

    # Collect post-test metrics;
    post_test_metrics = _collect_trainer_metrics(light_trainer)

    # Flush logger buffers (important for MLflow nested runs).
    for lg in light_trainer.loggers:
        if hasattr(lg, "finalize"):
            lg.finalize("success")

    overlapping_keys = set(post_test_metrics) & set(fit_metrics)
    if overlapping_keys:
        logger.debug("Preferring fit metrics over post-test metrics for keys: %s", sorted(overlapping_keys))

    # Keep fit metrics on key collisions (HPO uses val_* from fit).
    # This means Optuna will use e.g. val_dice (i.e. the fit-time value) for the monitor
    result_metrics = dict(post_test_metrics)
    result_metrics.update(fit_metrics)
    return result_metrics


def _run_post_fit_test(light_trainer: L.Trainer,
                       dataloaders: DataLoaders,
                       train_cfg: TrainConfig,
                       trainersetup: TrainerComponents,
                       light_model: L.LightningModule) -> None:
    # Get test or val dataloader depending on config
    test_loader = dataloaders.val if train_cfg.test_on_val_split else dataloaders.test
    if test_loader is None:
        raise RuntimeError("test_on_val_split is False but no test dataloader is available.")
    checkpoint_cb = next((cb for cb in trainersetup.callbacks if isinstance(cb, ModelCheckpoint)),
                         None)

    if train_cfg.use_checkpoint:
        if checkpoint_cb is None or not checkpoint_cb.best_model_path:
            raise RuntimeError("Testing requested with checkpoint enabled, but no best checkpoint is available")
        light_trainer.test(ckpt_path="best", dataloaders=test_loader)
    else:
        logger.warning("Checkpoint is disabled; testing with current in-memory weights instead.")
        light_trainer.test(light_model, dataloaders=test_loader)


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
