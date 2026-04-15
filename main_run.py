"""
Testing the AzureBlobMounter and loading a config from YAML, then setting up a dataloader for training.
"""


import argparse
import logging
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import lightning as L

from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml

from SkiNet.ML.dataloaders.dataloaders import RepeatDataLoader
from SkiNet.ML.transformations.plot_transformed_data import visualize_augmented_data
from SkiNet.ML.utils.model_factory import create_model
from SkiNet.ML.model.lightning_model import LightningModel


from SkiNet.Utils.logging.logging_callbacks_setup import setup_logging_and_callbacks
from SkiNet.ML.datasets.dataset_factory import create_segmentation_datasets_from_config
from SkiNet.ML.configs.experiment_config import ExperimentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SkiNet/ML/logging/system_metrics_callback.py


def configure_reproducibility(main_config: ExperimentConfig) -> bool | Literal['warn']:
    """
    Apply a single training seed across Lightning, PyTorch, and transform config defaults.
    """
    train_cfg = main_config.trainconfig
    L.seed_everything(train_cfg.seed, workers=True)

    # On Apple Silicon (MPS) Lightning raises MisconfigurationException if deterministic=True
    # because the MPS backend doesn't support deterministic mode. To avoid crashing at Trainer
    # construction when a developer runs on a Mac:
    #  - Preferred: set deterministic = "warn" so Lightning logs a warning but continues.
    #  - Alternative: set deterministic = False to silently disable determinism.
    # Note: True deterministic behavior requires CUDA/cuDNN. We only set cuDNN flags when CUDA is available.

    deterministic: bool | Literal['warn']  # for mypy
    if train_cfg.deterministic and torch.backends.mps.is_available():
        deterministic = "warn"  # use "warn" to avoid MisconfigurationException on MPS
    else:
        deterministic = train_cfg.deterministic

    if deterministic is True and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # fall back to train config seed if transform config seed is not set in config yaml
    if main_config.transformconfig.seed_value is None and "seed" not in main_config.transformconfig.compose_kwargs:
        main_config.transformconfig.seed_value = train_cfg.seed
    return deterministic


def create_dataloaders(main_config: ExperimentConfig) -> tuple[RepeatDataLoader, RepeatDataLoader, RepeatDataLoader]:
    """
    Build train/validation/test dataloaders from the experiment config.
    """
    train_cfg = main_config.trainconfig

    segm_datasets = create_segmentation_datasets_from_config(main_config)

    train_loader = RepeatDataLoader(segm_datasets.train,
                                    max_num_to_repeat=1,
                                    batch_size=train_cfg.batch_size,
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=train_cfg.num_workers)
    val_loader = RepeatDataLoader(segm_datasets.val,
                                  max_num_to_repeat=1,
                                  batch_size=train_cfg.batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=train_cfg.num_workers)

    test_loader = RepeatDataLoader(segm_datasets.test,
                                   max_num_to_repeat=1,
                                   batch_size=train_cfg.batch_size,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=train_cfg.num_workers)

    logging.info("Train dataset length: %d, Train DataLoader len (batches per epoch): %d",
                 len(segm_datasets.train), len(train_loader))
    return train_loader, val_loader, test_loader


def build_lightning_model(main_config: ExperimentConfig) -> LightningModel:
    """
    Build the Lightning segmentation model from the experiment config.
    """
    train_cfg = main_config.trainconfig

    model: torch.nn.Module = create_model(main_config)
    loss_fn = nn.BCEWithLogitsLoss()
    return LightningModel(model=model,
                          loss_fn=loss_fn,
                          num_classes=1,
                          lr=train_cfg.lr,
                          optimizer_name=train_cfg.optimizer_name,
                          weight_decay=train_cfg.weight_decay,
                          lr_scheduler_config=train_cfg.lr_scheduler_config)


def train_and_evaluate(main_config: ExperimentConfig, *, visualize: bool = True) -> dict[str, float]:
    """
    Run training and optional testing, then return epoch-level callback metrics as plain floats.
    """
    train_cfg = main_config.trainconfig
    deterministic = configure_reproducibility(main_config)
    train_loader, val_loader, test_loader = create_dataloaders(main_config)

    if visualize:
        visualize_augmented_data(dataset=train_loader.dataset, samples=20)
        # restore RNG state after visualization to avoid affecting training reproducibility
        L.seed_everything(train_cfg.seed, workers=True)

    light_model = build_lightning_model(main_config)

    light_model = build_lightning_model(main_config)
    trainersetup = setup_logging_and_callbacks(main_config=main_config)

    # Construct the Trainer using the resolved `deterministic` value instead of the
    # train_cfg.deterministic directly, to avoid potential MisconfigurationException on MPS when deterministic=True.
    light_trainer = L.Trainer(fast_dev_run=False,  # runs 1 train + 1 val batch only
                              max_epochs=train_cfg.max_epochs,
                              logger=trainersetup.loggers,
                              callbacks=trainersetup.callbacks,
                              accelerator=train_cfg.accelerator,
                              devices=train_cfg.devices,
                              log_every_n_steps=train_cfg.log_every_n_steps,
                              deterministic=deterministic)

    light_trainer.fit(light_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if train_cfg.run_test_after_fit:
        if train_cfg.test_on_val_split:
            light_trainer.test(light_model, dataloaders=val_loader)
        else:
            light_trainer.test(light_model, dataloaders=test_loader)

    metrics: dict[str, float] = {}
    for key, value in light_trainer.callback_metrics.items():
        if isinstance(value, torch.Tensor):
            metrics[key] = float(value.detach().cpu().item())
        elif isinstance(value, (float, int)):
            metrics[key] = float(value)
    return metrics


def main(cfg_path: Path) -> dict[str, float]:
    """
    Main function to run the pipeline from a config file.
    """
    main_config: ExperimentConfig = load_config_from_yaml(cfg_path)
    return train_and_evaluate(main_config)


if __name__ == "__main__":
    """
    Main entry point for the script.

    This script demonstrates how to use the AzureBlobMounter to mount Azure Blob Storage, load an experiment configuration from a YAML file,
    and set up a data loader for training.

    Example usage:
        python main_run.py --config path/to/config.yaml
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, help="Path to experiment YAML config")
    args = ap.parse_args()
    main(args.config)
