import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, cast

from pandas import DataFrame

from SkiNet.ML.configs.data_configs.base_data_config import BaseDataConfig
from SkiNet.Utils.experiment_keys import ExperimentType
from SkiNet.ML.datasets.segmentation_dataset import BaseDataset, SegmentationDataset
from SkiNet.ML.transformations.transform_data import get_transform_from_config
from SkiNet.ML.utils.model_utils import MLWorkflowState
from SkiNet.Utils.data.split_data import DataFrameSplits, split_segmentation_metadata
from SkiNet.ML.utils.typing_utils import TDataset_co
from SkiNet.ML.configs.experiment_config import ExperimentConfig
logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit(Generic[TDataset_co]):
    """
    Group of train/validation/test datasets created from one split operation.

    The container is generic so orchestration code can use the common shape while
    experiment-specific code can preserve concrete dataset types

    Attributes:
        train: Dataset for model training.
        val: Dataset for validation and early stopping.
        test: Dataset for final held-out evaluation.
        splits: Raw dataframe splits used to construct the train, val, test datasets.
    """
    train: TDataset_co
    val: TDataset_co
    test: TDataset_co
    splits: DataFrameSplits


def _split_by_predefined_column(df: DataFrame, column: str) -> DataFrameSplits:
    """
    Build DataFrameSplits from a column whose values are 'train', 'val', or 'test'.
    Rows with any other value are silently dropped.
    """
    import pandas as pd

    valid = {"train", "val", "test"}
    unknown = set(df[column].dropna().unique()) - valid
    if unknown:
        logger.warning("predefined_split_column '%s' contains unknown values %s — those rows will be skipped.", column, unknown)

    return DataFrameSplits(
        train=cast(pd.DataFrame, df[df[column] == "train"].copy()),
        val=cast(pd.DataFrame, df[df[column] == "val"].copy()),
        test=cast(pd.DataFrame, df[df[column] == "test"].copy()),
    )


class DatasetFactory(ABC, Generic[TDataset_co]):
    """
    Base class for experiment-specific dataset factories.

    A concrete factory is responsible for deriving the split dataframes,
    selecting the transforms for each workflow stage, and returning a typed
    ``DatasetSplit`` for the requested experiment type.
    """
    @abstractmethod
    def create_datasets(self, config: ExperimentConfig) -> DatasetSplit[TDataset_co]:
        """
        Build the full train/val/test dataset container for ``config``.

        Implementations are responsible for:
        - splitting the source metadata into train/val/test dataframes
        - resolving the appropriate transforms per split
        - constructing and returning a typed ``DatasetSplit``

        :param config: Fully resolved experiment configuration.
        :return: A ``DatasetSplit`` typed to this factory's dataset type.
        """
        pass


class SegmentationDatasetFactory(DatasetFactory[SegmentationDataset]):
    """
    Factory that creates ``SegmentationDataset`` objects for each workflow split.
    Use ``create_segmentation_datasets_from_config`` for the typed public entry point.
    """

    def create_datasets(self, config: ExperimentConfig) -> DatasetSplit[SegmentationDataset]:
        """
        Create the segmentation train/validation/test datasets for ``config``.

        :param config: Experiment configuration containing segmentation metadata,
            split configuration, transform configuration, and data root.
        :return: A ``DatasetSplit[SegmentationDataset]`` containing typed train, val,
            and test splits alongside the raw ``DataFrameSplits`` used to construct them.
        """
        data_config = cast(BaseDataConfig, config.dataconfig)
        metadata_df = data_config.metadata

        if data_config.predefined_split_column is not None:
            splits = _split_by_predefined_column(metadata_df, data_config.predefined_split_column)
        else:
            split_config = data_config.get_split_config()
            splits = split_segmentation_metadata(df=metadata_df, split_config=split_config)
        transformations = get_transform_from_config(config)
        cache_in_ram = config.trainconfig.cache_in_ram
        train_dataset = SegmentationDataset(config.dataconfig.data_root,
                                            splits.train,
                                            transformations.train,
                                            MLWorkflowState.TRAIN,
                                            cache_in_ram=cache_in_ram)
        val_dataset = SegmentationDataset(config.dataconfig.data_root,
                                          splits.val,
                                          transformations.val,
                                          MLWorkflowState.VAL,
                                          cache_in_ram=cache_in_ram)
        test_dataset = SegmentationDataset(config.dataconfig.data_root,
                                           splits.test,
                                           transformations.test,
                                           MLWorkflowState.TEST,
                                           cache_in_ram=cache_in_ram)

        # log basic info for observability
        try:
            logger.info(
                "Segmentation datasets created: train=%d, val=%d, test=%d",
                len(splits.train), len(splits.val), len(splits.test)
            )
        except Exception:
            logger.debug("Segmentation datasets created (counts unavailable)", exc_info=True)

        return DatasetSplit(train=train_dataset,
                            val=val_dataset,
                            test=test_dataset,
                            splits=splits)


# Registry mapping experiment types to their factories.
# Add new experiment types here as they are introduced.
_DATASET_FACTORIES: dict[ExperimentType, DatasetFactory[BaseDataset]] = {
    ExperimentType.SEGMENTATION: SegmentationDatasetFactory(),
}


def _get_dataset_factory(config: ExperimentConfig) -> DatasetFactory[BaseDataset]:
    """
    Resolve the factory for the experiment type in ``config``.
    To add a new experiment type, register its factory in ``_DATASET_FACTORIES``.

    :param config: Experiment configuration containing the experiment type.
    :return: The factory corresponding to ``config.experiment_type``.
    :raises ValueError: If the experiment type has no registered factory.
    """
    factory = _DATASET_FACTORIES.get(config.experiment_type)
    if factory is None:
        supported = ", ".join(str(k) for k in _DATASET_FACTORIES)
        raise ValueError(
            f"Unsupported experiment type '{config.experiment_type}'. "
            f"Supported: {supported}."
        )
    return factory


def create_segmentation_datasets_from_config(
    config: ExperimentConfig,
) -> DatasetSplit[SegmentationDataset]:
    """
    Typed entry point for segmentation experiments.

    Example::

        dataset_splits = create_segmentation_datasets_from_config(config)
        loader = DataLoader(dataset_splits.train, batch_size=8, shuffle=True)

    :param config: Experiment configuration for a segmentation experiment.
    :return: A ``DatasetSplit[SegmentationDataset]`` with typed train/val/test splits.
    """
    return SegmentationDatasetFactory().create_datasets(config)
