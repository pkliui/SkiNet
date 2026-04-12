import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast


from SkiNet.ML.configs.data_configs.base_data_config import BaseDataConfig
from SkiNet.Utils.experiment_keys import ExperimentType
from SkiNet.ML.configs.experiment_config import ExperimentConfig
from SkiNet.ML.datasets.segmentation_dataset import BaseDataset, SegmentationDataset
from SkiNet.ML.transformations.transform_data import get_transform_from_config
from SkiNet.ML.utils.model_utils import MLWorkflowState
from SkiNet.Utils.data.split_data import DataFrameSplits, split_segmentation_metadata


logger = logging.getLogger(__name__)


@dataclass
class DatasetBundle:
    """
    Generic container for datasets produced by a factory.
    """
    train: BaseDataset
    val: BaseDataset
    test: BaseDataset
    splits: DataFrameSplits


@dataclass
class SegmentationDatasets(DatasetBundle):
    train: SegmentationDataset
    val: SegmentationDataset
    test: SegmentationDataset
    splits: DataFrameSplits


class DatasetFactory(ABC):
    """
    Abstract base class for dataset factories. Each factory is responsible for creating datasets for a specific
    experiment type (e.g., segmentation, classification, etc.) based on the provided experiment configuration.
    """
    @abstractmethod
    def create_datasets(self, config: ExperimentConfig) -> DatasetBundle:
        pass

    def create_train_dataset(self, config: ExperimentConfig) -> BaseDataset:
        return self.create_datasets(config).train

    def create_val_dataset(self, config: ExperimentConfig) -> BaseDataset:
        return self.create_datasets(config).val

    def create_test_dataset(self, config: ExperimentConfig) -> BaseDataset:
        return self.create_datasets(config).test


class SegmentationDatasetFactory(DatasetFactory):
    def create_datasets(self, config: ExperimentConfig) -> DatasetBundle:
        """
        Create train, validation, and test datasets based on the provided experiment configuration.

        :param config: The experiment configuration containing dataset metadata, split configuration, and transformation configuration.
        :return: A DatasetBundle containing the created train, validation, and test datasets, along with the splits information.
        """
        data_config = cast(BaseDataConfig, config.dataconfig)
        metadata_df = data_config.metadata
        split_config = data_config.get_split_config()

        splits = split_segmentation_metadata(df=metadata_df,
                                             split_config=split_config)
        tranformation = get_transform_from_config(config)
        train_dataset = SegmentationDataset(config,
                                            splits.train,
                                            tranformation.train,
                                            MLWorkflowState.TRAIN)
        val_dataset = SegmentationDataset(config,
                                          splits.val,
                                          tranformation.val,
                                          MLWorkflowState.VAL)
        test_dataset = SegmentationDataset(config,
                                           splits.test,
                                           tranformation.test,
                                           MLWorkflowState.TEST)

        return SegmentationDatasets(train=train_dataset,
                                    val=val_dataset,
                                    test=test_dataset,
                                    splits=splits)


def _get_dataset_factory(config: ExperimentConfig) -> DatasetFactory:
    """
    Get the appropriate dataset factory based on the experiment type specified in the configuration.
    The returned factory can then be used to create datasets for the experiment.

    :param config: The experiment configuration containing the experiment type.
    :return: The dataset factory corresponding to the experiment type.
    """
    dataset_factories = {
        ExperimentType.SEGMENTATION: SegmentationDatasetFactory(),
    }
    exp_type = config.experiment_type
    if exp_type in dataset_factories:
        return dataset_factories[exp_type]
    else:
        raise ValueError(f"Unsupported experiment type: {exp_type}")


def create_pytorch_datasets_from_config(config: ExperimentConfig) -> DatasetBundle:
    """
    Factory method to create PyTorch datasets based on the experiment configuration.

    :param config: The experiment configuration containing dataset metadata, split configuration, and transformation configuration.
    :return: A DatasetBundle containing the created train, validation, and test datasets, along with the splits information.
    """
    factory = _get_dataset_factory(config)
    return factory.create_datasets(config)
