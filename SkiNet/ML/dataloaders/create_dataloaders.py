from dataclasses import dataclass
import logging
from SkiNet.ML.configs.train_configs.train_config import TrainConfig
from SkiNet.ML.utils.typing_utils import TDataset_co
from SkiNet.ML.configs.experiment_config import ExperimentConfig
from SkiNet.ML.dataloaders.dataloaders import RepeatDataLoader
from SkiNet.ML.datasets.dataset_factory import DatasetSplit, create_segmentation_datasets_from_config
from SkiNet.ML.datasets.segmentation_dataset import SegmentationDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataLoaders:
    """
    Container for the train, validation and test dataloaders used in the training and evaluation of the model.
    """
    train: RepeatDataLoader
    val: RepeatDataLoader
    test: RepeatDataLoader


def create_dataloaders_from_datasets(datasets: DatasetSplit[TDataset_co], train_cfg: TrainConfig) -> DataLoaders:
    """
    Generic builder — works for any dataset triple.

    :param datasets: DatasetSplit containing the train/val/test datasets to load from.
    :param train_cfg: TrainConfig containing dataloader parameters like batch size and num_workers.
    :return: DataLoaders containing the train/val/test dataloaders built from the provided datasets and config.
    """
    return DataLoaders(train=RepeatDataLoader(datasets.train, shuffle=True, max_num_to_repeat=1, batch_size=train_cfg.batch_size,
                                              num_workers=train_cfg.num_workers, drop_last=False,
                                              pin_memory=train_cfg.pin_memory,
                                              prefetch_factor=train_cfg.prefetch_factor if train_cfg.num_workers > 0 else None),
                       val=RepeatDataLoader(datasets.val, shuffle=False, max_num_to_repeat=1, batch_size=train_cfg.batch_size,
                                            num_workers=train_cfg.num_workers, drop_last=False,
                                            pin_memory=train_cfg.pin_memory,
                                            prefetch_factor=train_cfg.prefetch_factor if train_cfg.num_workers > 0 else None),
                       test=RepeatDataLoader(datasets.test, shuffle=False, max_num_to_repeat=1, batch_size=train_cfg.batch_size,
                                             num_workers=train_cfg.num_workers, drop_last=False,
                                             pin_memory=train_cfg.pin_memory,
                                             prefetch_factor=train_cfg.prefetch_factor if train_cfg.num_workers > 0 else None))


def create_segmentation_dataloaders(main_config: ExperimentConfig) -> DataLoaders:
    """
    Segmentation-specific entry point for creating dataloaders

    :param main_config: Main configuration read from YAML file
    :return Dataloaders class whose fields are train, val and test dataloaders
    """
    segm_datasets: DatasetSplit[SegmentationDataset] = create_segmentation_datasets_from_config(main_config)
    loaders = create_dataloaders_from_datasets(segm_datasets, main_config.trainconfig)
    logger.info("Train dataset length: %d, batches per epoch: %d",
                len(segm_datasets.train), len(loaders.train))
    return loaders
