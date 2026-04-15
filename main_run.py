"""
Testing the AzureBlobMounter and loading a config from YAML, then setting up a dataloader for training.
"""


import argparse
import logging
from pathlib import Path

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


def main(cfg_path: Path) -> None:
    """
    Main function to run the pipeline.
    """

    # Main config
    main_config: ExperimentConfig = load_config_from_yaml(cfg_path)
    train_cfg = main_config.trainconfig

    # Datasets
    segm_datasets = create_segmentation_datasets_from_config(main_config)
    visualize_augmented_data(dataset=segm_datasets.train, samples=20)

    # Dataloaders
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

    # Lightning model
    model: torch.nn.Module = create_model(main_config)
    loss_fn = nn.BCEWithLogitsLoss()
    light_model = LightningModel(model=model,
                                 loss_fn=loss_fn,
                                 num_classes=1,
                                 lr=train_cfg.lr,
                                 optimizer_name=train_cfg.optimizer_name,
                                 weight_decay=train_cfg.weight_decay,
                                 lr_scheduler_config=train_cfg.lr_scheduler_config)

    # Lightning trainer
    trainersetup = setup_logging_and_callbacks(main_config=main_config)
    light_trainer = L.Trainer(fast_dev_run=False,  # runs 1 train + 1 val batch only
                              max_epochs=train_cfg.max_epochs,
                              logger=trainersetup.loggers,
                              callbacks=trainersetup.callbacks,
                              accelerator=train_cfg.accelerator,
                              devices=train_cfg.devices,
                              log_every_n_steps=train_cfg.log_every_n_steps)

    light_trainer.fit(light_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if train_cfg.run_test_after_fit:
        if train_cfg.test_on_val_split:
            light_trainer.test(light_model, dataloaders=val_loader)
        else:
            light_trainer.test(light_model, dataloaders=test_loader)


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
