import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from SkiNet.ML.configs.experiment_config import ExperimentConfig
from SkiNet.ML.datasets.sample_specs import Sample, create_valid_samplespecs, load_sample
from SkiNet.ML.transformations.transform_adapters import SampleTransformAdapter

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
                 config: ExperimentConfig,
                 transform: SampleTransformAdapter) -> None:
        """
        :param config: The experiment configuration containing dataset metadata and data root information.
        """
        super().__init__(config=config)
        self.config = config
        self.dataframe = config.dataconfig.metadata
        """A pandas DataFrame containing metadata for the dataset."""
        self.data_root = Path(config.dataconfig.data_root)
        """Data root path where images and masks are stored, derived from the experiment configuration."""
        logger.debug("Data root in SegmentationDataset: %s", self.data_root)
        self.sample_specs = create_valid_samplespecs(self.dataframe)
        """A dictionary containing the valid sample specifications, derived from the DataFrame, such as image and mask paths and metadata."""
        self.sample_ids = list(self.sample_specs.keys())
        """A list of sample IDs corresponding to the valid samples in the dataset, derived from the sample specifications."""
        self.transform = transform
        logger.info("SegmentationDataset initialized with transform=%r", self.transform)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get_sample_item(index)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def get_raw_sample(self, index: int) -> Sample:
        """
        Load a raw sample from disk without applying any transforms.
        Useful for visualization and debugging without mutating dataset.transform.
        """
        specs_item = self.sample_specs[self.sample_ids[index]]
        return load_sample(specs_item, data_root=self.data_root)  # image and mask should be CHW, uint8

    def get_sample_item(self, index: int) -> dict[str, Any]:
        """
        Get a single sample item by index.

        :return: A dictionary containing the image tensor, mask tensor, and sample specifications for the specified index.
        """
        specs_item = self.sample_specs[self.sample_ids[index]]
        sample = load_sample(specs_item,
                             data_root=self.data_root)  # image and mask should be CHW, uint8

        transformed_sample = self.transform(sample=sample)

        image_tensor = transformed_sample.image
        mask_tensor = transformed_sample.mask

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "specs": sample.specs.model_dump(),
        }
