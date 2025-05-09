"""Various functions to load images and masks"""
import logging
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from torch import randint
from torch.utils.data.dataset import Dataset


def get_random_sample(data_set: Dataset) -> dict[np.array, np.array]:
    """
    Function to return a random pair of an image and a mask having the same sample_idx

    :return dictionary containing an image array and a mask array randomly selected from the provided dataset

    """
    sample_idx = randint(len(data_set), size=(1,)).item()

    img = data_set[sample_idx]['image']
    mask = data_set[sample_idx]['mask']
    sample_name = Path(data_set.images_list[sample_idx]).parent.parent.name

    return {'image': img, 'mask': mask, 'name': sample_name}


def read_images_from_directory(directory_path: str, 
                               search_pattern: str, 
                               max_num_images_to_return: int = 1) -> List[Image.Image]:
    """
    Reads all images in the directory that match the given search pattern.

    :param directory_path: Path to the directory containing images.
    :param search_pattern: For example, for BMP images located in a folder having "Dermoscopic_Image" in its name, use '*_Dermoscopic_Image/*.bmp'.
    :param max_num_images_to_return: Number of images to read. Default is 1.

    :return: A list of PIL Image objects.
    """
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


def read_paths_to_images_from_directory(directory_path, search_pattern) -> list[Path]:
    """
    Reads paths to all images in the directory that match the given search pattern

    :param directory_path: Path to the directory containing images.
    :param search_pattern: for example for bmp images located in a folder  having "Dermoscopic_Image" in its name '*_Dermoscopic_Image/*.bmp'
    :return: A list of paths to images.
    """
    image_paths = list(Path(directory_path).rglob(search_pattern))
    
    return image_paths

