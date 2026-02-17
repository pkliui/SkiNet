"""
Contains various functions to handle data
"""
import logging
import re
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def extract_sample_number(file_path: Union[Path, str]) -> int:
    """
    Extracts a numeric sample identifier from a file name given its full path

    For example, if the file path is "path_to_image/image_03.bmp", it will return 3.
    Otherwise, if no digits found, it will return 0.

    :param file_path: full path to file, e.g. path_to_file/in_a_folder/the_file.bmp provided as a string or Path
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)  # needed for Azure as the path is a str

    # Get the filename without the extension
    file_stem = file_path.stem
    # Find the first group of digits in the filename (we assume they are the only one)
    first_matching_digits = re.search(r'\d+', file_stem)
    return int(first_matching_digits.group()) if first_matching_digits else 0

def filter_and_pair_valid_paths(paths_to_images: List[Path],
                                paths_to_masks: List[Path],
                                filter_if_size_different: bool) -> Tuple[List[Path], List[Path]]:
    """
    Filters and pairs image and mask paths so that:
    - Only pairs with both image and mask present are kept
    - Only pairs where image and mask have the same size are kept, if filter_if_size_different is True.

    :param paths_to_images: List of image file paths
    :param paths_to_masks: List of mask file paths
    :param filter_if_size_different: If True, filter out pairs where images and masks do not have the same size
    :return: Two lists: filtered and paired image paths and mask paths
    """
    # filter images and masks that do not have a counterpart
    paths_to_images, paths_to_masks = filter_missing_images_and_masks(paths_to_images, paths_to_masks)
    if filter_if_size_different:
        # filter images and masks that do not have the same size
        paths_to_images, paths_to_masks = filter_images_and_masks_of_different_sizes(paths_to_images, paths_to_masks)

    return paths_to_images, paths_to_masks


def filter_missing_images_and_masks(paths_to_images: List[Path], paths_to_masks: List[Path]) -> Tuple[List[Path], List[Path]]:
    """
    Given paths to images and masks, identify pairs of those based on a unique sample number extracted from the filename.
    Modify the provided paths by including only image and masks that have the same sample number, i.e. have a pair

    :param paths_to_images: List of image file paths, not necessarily sorted
    :param paths_to_masks: List of mask file paths, not necessarily sorted
    :return: Two lists: image paths and mask paths, in which all items are uniquely pairable by their sample number, sorted
    """
    # Create dictionaries keyed by the sample number and identify common keys
    image_dict = {extract_sample_number(img): img for img in paths_to_images}
    mask_dict = {extract_sample_number(msk): msk for msk in paths_to_masks}
    common_keys = set(image_dict.keys()) & set(mask_dict.keys())

    # Identify missing items
    missing_images = set(mask_dict.keys()) - set(image_dict.keys())
    missing_masks = set(image_dict.keys()) - set(mask_dict.keys())
    for key in missing_images:
        logging.getLogger(__name__).info(f"Missing image for sample number {key}, mask with the same sample number will not be included into dataset.")
    for key in missing_masks:
        logging.getLogger(__name__).info(f"Missing mask for sample number {key}, image with the same sample number will not be included into dataset.")

    # Update the input lists using the common keys and sort them
    paired_images = [image_dict[key] for key in sorted(common_keys)]
    paired_masks = [mask_dict[key] for key in sorted(common_keys)]
    return paired_images, paired_masks


def filter_images_and_masks_of_different_sizes(paths_to_images: List[Path], paths_to_masks: List[Path]) -> Tuple[List[Path], List[Path]]:
    """
    Filter out images and masks that do not have the same size.

    :param paths_to_images: List of image file paths
    :param paths_to_masks: List of mask file paths
    :return: Two lists: filtered image paths and filtered mask paths
    """
    filtered_images_paths = []
    filtered_masks_paths = []

    for img_path, msk_path in zip(paths_to_images, paths_to_masks):
        img = Image.open(img_path)
        msk = Image.open(msk_path)

        if img.size == msk.size:
            filtered_images_paths.append(img_path)
            filtered_masks_paths.append(msk_path)
        else:
            logging.getLogger(__name__).info(f"Image {img_path} and mask {msk_path} do not have the same size, skipping this pair.")

    return filtered_images_paths, filtered_masks_paths

def convert_to_numpy_bytes(image_paths: list[Path], mask_paths: list[Path]) -> Tuple[NDArray[np.bytes_], NDArray[np.bytes_]]:
    """
    Convert lists of image and mask paths to NumPy arrays of bytes.

    :param image_paths: List of image file paths
    :param mask_paths: List of mask file paths
    :return: Two NumPy arrays: one for image paths and one for mask paths
    """
    image_paths_array = np.array(image_paths, dtype=np.bytes_)
    mask_paths_array = np.array(mask_paths, dtype=np.bytes_)
    assert isinstance(image_paths_array, np.ndarray) and image_paths_array.dtype == np.bytes_
    assert isinstance(mask_paths_array, np.ndarray) and mask_paths_array.dtype == np.bytes_
    return image_paths_array, mask_paths_array
