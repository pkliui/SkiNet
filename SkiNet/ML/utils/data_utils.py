"""
Contains various functions to handle data
"""
from pathlib import Path
import re
from typing import List, Tuple
import logging


def extract_sample_number(file_path: Path) -> int:
    """
    Extracts a numeric sample identifier from a file name given its full path
    
    For example, if the file path is "path_to_image/image_03.bmp", it will return 3.
    Otherwise, if no digits found, it will return 0.

    :param file_path: full path to file, e.g. path_to_file/in_a_folder/the_file.bmp
    """
    # Get the filename without the extension
    file_stem = file_path.stem
    # Find the first group of digits in the filename (we assume they are the only one)
    first_matching_digits = re.search(r'\d+', file_stem)
    return int(first_matching_digits.group()) if first_matching_digits else 0


def filter_missing_images_and_masks(image_paths: List[Path], mask_paths: List[Path]) -> Tuple[List[Path], List[Path]]:
    """
    Given paths to images and masks, identify pairs of those based on a unique sample number extracted from the filename.
    Modify the provided paths by including only image and masks that have the same sample number, i.e. have a pair
    
    :param image_paths: List of image file paths, not necessarily sorted
    :param mask_paths: List of mask file paths, not necessarily sorted
    :return: Two lists: image paths and mask paths, in which all items are uniquely pairable by their sample number, sorted
    """
    # Create dictionaries keyed by the sample number and identify common keys
    image_dict = {extract_sample_number(img): img for img in image_paths}
    mask_dict = {extract_sample_number(msk): msk for msk in mask_paths}
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
    