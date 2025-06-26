"""Various functions to load images and masks"""
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from torch import randint
from torch.utils.data.dataset import Dataset

from SkiNet.ML.utils.data_utils import filter_and_pair_valid_paths


def get_random_sample(data_set: Dataset) -> dict[np.array, np.array]:
    """
    Function to return a random pair of an image and a mask having the same sample_idx

    :return dictionary containing an image array and a mask array randomly selected from the provided dataset

    """
    sample_idx = randint(len(data_set), size=(1,)).item()

    img = data_set[sample_idx]['image']
    mask = data_set[sample_idx]['mask']
    sample_name = Path(data_set.images_list[sample_idx].decode('utf-8')).parent.parent.name

    return {'image': img, 'mask': mask, 'name': sample_name}


def read_images_from_directory(directory_path: str, 
                               search_pattern: str, 
                               max_num_images_to_return: int = 1) -> List[Image.Image]:
    """
    Reads all images in the directory that match the given search pattern.
    Was replaced by read_images_and_masks_from_directory, but still can be used as stand-alone where no filtering is required.

    :param directory_path: Path to the directory containing images.
    :param search_pattern: For example, for BMP images located in a folder having "Dermoscopic_Image" in its name, use '*_Dermoscopic_Image/*.bmp'.
    :param max_num_images_to_return: Number of images to read. Default is 1.

    :return: A list of PIL Image objects.
    """
    logging.getLogger(__name__).debug("read_images_from_directory is HERE  ")

    # Use rglob to find all files matching the search pattern
    image_paths = list(Path(directory_path).rglob(search_pattern))
    # List to store PIL Image objects
    images: List[Image.Image] = []
    # Loop through the paths and open the images
    for image_path in image_paths[:max_num_images_to_return]:
        try:
            # Open image using PIL and append to list
            img = Image.open(image_path)
            images.append(img)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not open image {image_path}: {e}")
    return images

def read_images_and_masks_from_directory(directory_path: str, 
                                search_pattern_images: str, 
                                search_pattern_masks: str,
                                max_num_images_to_return: int = 1) -> Tuple[List[Image.Image], List[Image.Image]]:
    """
    Reads all images and masks in the directory that match the given search pattern. Filter out pairs where images and masks are not present or have different sizes.
    Replaces read_images_from_directory where no filtering was done.

    :param directory_path: Path to the root directory containing images and masks
    :param search_pattern_images: For example, for BMP images located in a folder having "Dermoscopic_Image" in its name, use '*_Dermoscopic_Image/*.bmp'.
    :param search_pattern_masks: For example, for BMP masks located in a folder having "lesion" in its name, use '*_lesion/*.bmp'.
    :param max_num_images_to_return: Number of images to read. Default is 1.

    :return: A tuple of images and masks as lists of PIL Image objects.
    """
    logging.getLogger(__name__).debug("read_images_and_masks_from_directory is HERE  ")

    # Use rglob to find all files matching the search pattern
    images_paths = list(Path(directory_path).rglob(search_pattern_images))
    masks_paths = list(Path(directory_path).rglob(search_pattern_masks))

    # filter missing pairs or pairs where images and masks are of different size
    images_paths, masks_paths = filter_and_pair_valid_paths(images_paths, masks_paths)

    # List to store PIL Image objects
    images: List[Image.Image] = []
    masks: List[Image.Image] = []

    # Loop through the paths and open the images
    for image_path, mask_path in zip(images_paths[:max_num_images_to_return], masks_paths[:max_num_images_to_return]):
        try:
            # Open image using PIL and append to list
            img = Image.open(image_path)
            images.append(img)
            # Open mask using PIL and append to list
            mask = Image.open(mask_path)
            masks.append(mask)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not open image {image_path} or mask {mask_path}: {e}")
    return images, masks


def read_paths_to_images_from_directory(directory_path, search_pattern) -> list[Path]:
    """
    Reads paths to all images in the directory that match the given search pattern

    :param directory_path: Path to the directory containing images.
    :param search_pattern: for example for bmp images located in a folder  having "Dermoscopic_Image" in its name '*_Dermoscopic_Image/*.bmp'
    :return: A list of paths to images.
    """
    image_paths = list(Path(directory_path).rglob(search_pattern))
    
    return image_paths

