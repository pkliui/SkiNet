from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field
from torchvision.io import decode_image

from SkiNet.Utils.csv_headers import DATAPATH_HEADER, DATATYPE_HEADER, DATATYPE_IMAGE, DATATYPE_MASK, SAMPLEID_HEADER

logger = logging.getLogger(__name__)


class SampleSpecs(BaseModel):
    """
    Represents basic specs container for a sample consisting of an image and a mask

    :attributes:
    - image_path: Path to the image file
    - mask_path: Path to the mask file
    - metadata: Additional metadata associated with the sample
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample_id: str
    image_path: str
    mask_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Sample(BaseModel):
    """
    Represents a single training sample consisting of an image and a mask.

    :attributes:
    - image: The image tensor for the sample or, after augmentation, a numpy array that can be converted to a tensor.
    - mask: The mask tensor for the sample or, after augmentation, a numpy array that can be converted to a tensor.
    - specs: The SampleSpecs object containing metadata and paths for the sample.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Union[torch.Tensor, np.ndarray]
    mask: Union[torch.Tensor, np.ndarray]
    specs: SampleSpecs


def load_data_item(item_rel_path: str,
                   data_root: Path) -> torch.Tensor:
    """
    Load a single data item (image or mask) from the specified data root and relative path, and return it as a tensor.

    :param item_rel_path: Path to the data item relative to data root
    :param data_root: Root directory location, it is assumed to be a local path.

    :return A torch.Tensor containing the loaded image or mask data.
      - shape: (C, H, W)
      - dtype: torch.uint8
    """
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: '{data_root}'")

    full_path = data_root / item_rel_path
    if not full_path.exists():
        raise FileNotFoundError(f"Data item not found: '{full_path}'")

    image: torch.Tensor = decode_image(str(full_path))  # CHW, usually uint8

    if image.dtype != torch.uint8:
        image = image.to(torch.uint8)

    return image  # CHW, uint8


def load_sample(specs: SampleSpecs,
                data_root: Path) -> Sample:
    """
    Load a single training sample consisting of an image and a mask.

    :param specs: SampleSpecs object containing metadata and paths for the sample.
    :param data_root: The root directory where the data is stored, it is assumed to be a local path.

    :return: A Sample object containing the loaded image and mask tensors (CHW, uint8), along with the sample specifications.
    """
    image = load_data_item(specs.image_path, data_root=data_root)  # torch.Tensor CHW uint8
    mask = load_data_item(specs.mask_path, data_root=data_root)  # torch.Tensor CHW uint8

    return Sample(image=image, mask=mask, specs=specs)  # CHW, uint8


def create_valid_samplespecs(df: pd.DataFrame, preserve_original_order: bool = False) -> dict[str, SampleSpecs]:
    """
    Create a dictionary of valid SampleSpecs objects from a DataFrame.

    For each unique sample_id in the DataFrame, this function extracts the corresponding image and mask paths,
    along with any additional metadata and returns a dictionary mapping sample_id to SampleSpecs.

    Note that only sample_ids that have both an image and a mask will be included in the resulting dictionary.
    If there are multiple images or masks for a sample_id, a warning will be logged and only the first one will be used.
    If there is a metadata mismatch between the image and mask rows for a sample_id, a warning will be logged and that sample_id will be skipped.

    :param df: DataFrame containing columns for sample_id, data_type (image/mask), data_root, and any additional metadata.
    :return: Dictionary mapping sample_id to SampleSpecs objects.
    """
    sample_specs = {}

    if preserve_original_order:
        groups = df.groupby(SAMPLEID_HEADER, sort=False)
    else:
        # Sort by sample_id to ensure consistent ordering
        groups = df.sort_values(by=SAMPLEID_HEADER).groupby(SAMPLEID_HEADER)

    for sample_id, group in groups:
        # get all rows containing images for the current sample id
        image_rows = group[group[DATATYPE_HEADER] == DATATYPE_IMAGE]
        # get all rows containing masks for the current sample id
        mask_rows = group[group[DATATYPE_HEADER] == DATATYPE_MASK]

        if len(image_rows) != 0 and len(mask_rows) != 0:
            if len(image_rows) > 1:
                logger.warning(f"Multiple images found for sample_id {sample_id}. Using the first one.")
            if len(mask_rows) > 1:
                logger.warning(f"Multiple masks found for sample_id {sample_id}. Using the first one.")

            # nevertheless, pick the first image and mask path
            image_path = image_rows[DATAPATH_HEADER].iloc[0]
            mask_path = mask_rows[DATAPATH_HEADER].iloc[0]

            # extract additional metadata columns
            metadata_columns = [col for col in df.columns if col not in [
                SAMPLEID_HEADER, DATAPATH_HEADER, DATATYPE_HEADER]]

            image_meta_df = image_rows[metadata_columns].iloc[0]
            mask_meta_df = mask_rows[metadata_columns].iloc[0]

            if not image_meta_df.equals(mask_meta_df):
                logger.warning(f"Metadata mismatch for sample_id {sample_id}. Skipping sample.")
            else:
                # create SampleSpecs object and add to dictionary
                # only for unique sample ids with both image and mask present and matching metadata
                sample_specs[sample_id] = SampleSpecs(sample_id=sample_id,
                                                      image_path=image_path,
                                                      mask_path=mask_path,
                                                      metadata=image_meta_df.to_dict())
        else:
            if len(image_rows) == 0:
                logger.warning(f"No image found for sample_id {sample_id}")
            if len(mask_rows) == 0:
                logger.warning(f"No mask found for sample_id {sample_id}")

    return sample_specs
