import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from SkiNet.ML.configs.experiment_config import ExperimentConfig
from SkiNet.ML.datasets.sample_specs import create_valid_samplespecs, load_sample
from SkiNet.ML.transformations.crop_data import crop_2d_image

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
        self.config = config
        self.dataframe = config.dataconfig.metadata
        """A pandas DataFrame containing metadata for the dataset."""
        self.data_root = Path(config.dataconfig.data_root)
        """Data root path where images and masks are stored, derived from the experiment configuration."""
        logger.debug("Data root in SegmentationDataset: %s", self.data_root)
        self.sample_specs = create_valid_samplespecs(self.dataframe)
        """A dictionary containing the valid sample specifications."""
        self.sample_ids = list(self.sample_specs.keys())
        """A list of sample IDs corresponding to the valid samples in the dataset, derived from the sample specifications."""

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
        # Apply cropping to the image and mask tensors based on the specified crop size in the experiment configuration.
        slices_image = crop_2d_image(sample.image, self.config.dataconfig.crop_size)
        slices_mask = crop_2d_image(sample.mask, self.config.dataconfig.crop_size)

        return {"image": sample.image[slices_image], "mask": sample.mask[slices_mask], "specs": sample.specs.model_dump()}
