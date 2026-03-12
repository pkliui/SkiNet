import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from SkiNet.ML.configs.experiment_config import ExperimentConfig
from SkiNet.ML.datasets.sample_specs import create_valid_samplespecs, load_sample

logger = logging.getLogger(__name__)

class BaseDataset(Dataset, ABC):
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        raise NotImplementedError("Subclasses must implement __getitem__ method")

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError("Subclasses must implement __len__ method")


class SegmentationDataset(BaseDataset):
    """
    Dataset for semantic segmentation tasks.
    """

    def __init__(self,
                 config: ExperimentConfig) -> None:
        """
        :param config: The experiment configuration containing dataset metadata and data root information.
        """
        super().__init__(config=config)

        # get the dataframe and the data root from config
        self.dataframe = config.dataconfig.metadata
        logger.debug("Dataframe used for SegmentationDataset %s: %s", self.dataframe.head())

        local_root = config.dataconfig.local_data_root
        if local_root is None:
            raise ValueError("Local data root must be specified in the experiment configuration for SegmentationDataset")

        self.data_root = Path(local_root)
        logger.debug("Data root in SegmentationDataset: %s", self.data_root)

        # get the sample specifications for pairs of images and masks,

        self.sample_specs = create_valid_samplespecs(self.dataframe)
        self.sample_ids = list(self.sample_specs.keys())

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get_sample_item(index)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def get_sample_item(self, index: int) -> dict[str, Any]:
        """
        Get a single sample item by index.

        :return: A dictionary containing the image tensor, mask tensor, and sample specifications for the specified index.
        """
        specs_item = self.sample_specs[self.sample_ids[index]]
        sample = load_sample(specs_item,
                             data_root=self.data_root)
        return {"image": sample.image, "mask": sample.mask, "specs": sample.specs.model_dump()}
