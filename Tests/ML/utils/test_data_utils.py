import pytest
from pathlib import Path
import logging
import tempfile
from PIL import Image
import numpy as np
from typing import List
import random
from SkiNet.ML.utils.data_utils import extract_sample_number, filter_missing_images_and_masks

IMG_SIZE = 128
NUM_SAMPLES = 7

"""Testing extract_sample_number"""

@pytest.mark.parametrize("filename, expected", [
    ("IMD001.bmp", 1), # should return 1, not 001
    ("sample_42_mask.png", 42), # should return 42
    ("test123.bmp", 123), # should return 123
    ("no_number_here.bmp", 0),  # No digits should return 0
    ("random_99_test_1.tiff", 99),  # Takes first number found, should return 99
    ("100_sample11.tif", 100),  # 100 is at start, so should be extracted, not 11
    ("img.bmp", 0),  # No digits should return 0
])
def test_extract_sample_number(filename, expected):
    """
    Test that extract_sample_number correctly extracts numbers from filenames given assertions above
    """
    file_path = Path(filename)
    assert extract_sample_number(file_path) == expected


"""Testing local_ph2dataset"""

@pytest.fixture
def local_ph2dataset_one_sample_each_missing(tmp_path):
    """
    Create local_ph2dataset with missing image and mask samples
    """
    def _local_ph2dataset_one_sample_each_missing(image_sample_num_to_skip: List[int], mask_sample_num_to_skip: List[int]):
        """
        :param image_sample_num_to_skip: image indices to skip
        :param mask_sample_num_to_skip: mask indices to skip

        :return path to the root folder containing images and masks
        """
        for ii in range(NUM_SAMPLES):
            if ii not in image_sample_num_to_skip:
                img_folder = tmp_path / f"IMD{ii}" / f"IMD{ii}_Dermoscopic_Image"
                img_folder.mkdir(parents=True, exist_ok=True)
                img = Image.fromarray(np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE), dtype=np.uint8))
                img.save(img_folder / f"IMD{ii}.bmp")
            else:
                logging.getLogger(__name__).info(f"Skipping image sample {ii} while creating local_ph2dataset for testing.")

            if ii not in mask_sample_num_to_skip:
                mask_folder = tmp_path / f"IMD{ii}" / f"IMD{ii}_lesion"
                mask_folder.mkdir(parents=True, exist_ok=True)
                mask = Image.fromarray(np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE), dtype=np.uint8))
                mask.save(mask_folder / f"IMD{ii}_lesion.bmp")
            else:
                logging.info(f"Skipping mask sample {ii} while creating local_ph2dataset for testing.")
        return tmp_path

    yield _local_ph2dataset_one_sample_each_missing


@pytest.mark.parametrize(["image_sample_num_to_skip", "mask_sample_num_to_skip"],
                         [([2], [3])])
def test_pair_image_mask_paths_one_sample_missing(local_ph2dataset_one_sample_each_missing, image_sample_num_to_skip, mask_sample_num_to_skip):
    """
    Test that the returned paths to images and masks include only those that have the same sample number 
    (i.e. that "image_sample_num_to_skip", "mask_sample_num_to_skip" are not included) and are sorted according to their sample number
    Use local_ph2dataset with missing one image and missing one mask samples
    """

    # get the path to the root folder containing images and masks
    dataset_root = local_ph2dataset_one_sample_each_missing(image_sample_num_to_skip, mask_sample_num_to_skip)
    # given the root folder, create expected paths to images and masks
    # image 2 missing
    expected_images = [
        dataset_root / f"IMD0" / f"IMD0_Dermoscopic_Image" / f"IMD0.bmp",
        dataset_root / f"IMD1" / f"IMD1_Dermoscopic_Image" / f"IMD1.bmp",
        dataset_root / f"IMD4" / f"IMD4_Dermoscopic_Image" / f"IMD4.bmp",
        dataset_root / f"IMD5" / f"IMD5_Dermoscopic_Image" / f"IMD5.bmp",
        dataset_root / f"IMD6" / f"IMD6_Dermoscopic_Image" / f"IMD6.bmp"
    ]

    #  mask 3 missing
    expected_masks = [
        dataset_root / f"IMD0" / f"IMD0_lesion" / f"IMD0_lesion.bmp",
        dataset_root / f"IMD1" / f"IMD1_lesion" / f"IMD1_lesion.bmp",
        dataset_root / f"IMD4" / f"IMD4_lesion" / f"IMD4_lesion.bmp",
        dataset_root / f"IMD5" / f"IMD5_lesion" / f"IMD5_lesion.bmp",
        dataset_root / f"IMD6" / f"IMD6_lesion" / f"IMD6_lesion.bmp"
    ]

    # get the actual paths to images and masks located in the root folder
    path_to_images = list(Path(dataset_root).rglob('*_Dermoscopic_Image/*.bmp'))
    path_to_masks = list(Path(dataset_root).rglob('*_lesion/*.bmp'))

    # apply filter_missing_images_and_masks and check the results    
    images, masks = filter_missing_images_and_masks(path_to_images, path_to_masks)
    assert images == expected_images, f"Paired images do not match expected paths: {images} vs {expected_images}"
    assert masks == expected_masks, f"Paired masks do not match expected paths: {masks} vs {expected_masks}"




@pytest.fixture
def local_ph2dataset_one_sample_each_missing_unsorted(tmp_path):
    """
    Create local_ph2dataset with missing image and mask samples.
    Assume random order both for images and masks, i.e. samples cannot be paired (the first in the list of paths yiels image 0 and mask 1, for example)
    """
    def _local_ph2dataset_one_sample_each_missing_unsorted(image_sample_num_to_skip: List[int], mask_sample_num_to_skip: List[int]):
        """
        :param image_sample_num_to_skip: image indices to skip
        :param mask_sample_num_to_skip: mask indices to skip

        :return path to the root folder containing images and masks
        """
        # create ordered sample indices 
        sample_indices = list(range(NUM_SAMPLES))
        #
        # lets assume the paths to images are ordered
        for ii in sample_indices:
            if ii not in image_sample_num_to_skip:
                img_folder = tmp_path / f"IMD{ii}" / f"IMD{ii}_Dermoscopic_Image"
                img_folder.mkdir(parents=True, exist_ok=True)
                img = Image.fromarray(np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE), dtype=np.uint8))
                img.save(img_folder / f"IMD{ii}.bmp")
            else:
                logging.info(f"Skipping image sample {ii} while creating local_ph2dataset for testing.")

        # shuffle the indices of masks
        random.shuffle(sample_indices)

        # now create  masks with these indices
        for ii in sample_indices:
            if ii not in mask_sample_num_to_skip:
                mask_folder = tmp_path / f"IMD{ii}" / f"IMD{ii}_lesion"
                mask_folder.mkdir(parents=True, exist_ok=True)
                mask = Image.fromarray(np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE), dtype=np.uint8))
                mask.save(mask_folder / f"IMD{ii}_lesion.bmp")
            else:
                logging.info(f"Skipping mask sample {ii} while creating local_ph2dataset for testing.")
        return tmp_path

    yield _local_ph2dataset_one_sample_each_missing_unsorted


@pytest.mark.parametrize(["image_sample_num_to_skip", "mask_sample_num_to_skip"],
                         [([2], [3])])
def test_pair_image_mask_paths_one_sample_missing_unsorted(local_ph2dataset_one_sample_each_missing_unsorted, image_sample_num_to_skip, mask_sample_num_to_skip):
    """
    Test that the returned paths to images and masks include only those that have the same sample number 
    (i.e. that "image_sample_num_to_skip", "mask_sample_num_to_skip" are not included) and can be correctly paired, i.e. image 0 - mask 0
    
    Use local_ph2dataset with missing one image and missing one mask samples, where the order of masks has been shuffled
    """
    # get the path to the root folder containing images and masks
    dataset_root = local_ph2dataset_one_sample_each_missing_unsorted(image_sample_num_to_skip, mask_sample_num_to_skip)

    # given the root folder, create expected paths to images and masks
    # image 2 missing
    expected_images = [
        dataset_root / f"IMD0" / f"IMD0_Dermoscopic_Image" / f"IMD0.bmp",
        dataset_root / f"IMD1" / f"IMD1_Dermoscopic_Image" / f"IMD1.bmp",
        dataset_root / f"IMD4" / f"IMD4_Dermoscopic_Image" / f"IMD4.bmp",
        dataset_root / f"IMD5" / f"IMD5_Dermoscopic_Image" / f"IMD5.bmp",
        dataset_root / f"IMD6" / f"IMD6_Dermoscopic_Image" / f"IMD6.bmp"
    ]

    #  mask 3 missing - we expect the list of paths to masks to be ordered after applying filter_missing_images_and_masks
    expected_masks = [
        dataset_root / f"IMD0" / f"IMD0_lesion" / f"IMD0_lesion.bmp",
        dataset_root / f"IMD1" / f"IMD1_lesion" / f"IMD1_lesion.bmp",
        dataset_root / f"IMD4" / f"IMD4_lesion" / f"IMD4_lesion.bmp",
        dataset_root / f"IMD5" / f"IMD5_lesion" / f"IMD5_lesion.bmp",
        dataset_root / f"IMD6" / f"IMD6_lesion" / f"IMD6_lesion.bmp"
    ]

    #get the actual paths to images and masks located in the root folder
    path_to_images = list(Path(dataset_root).rglob('*_Dermoscopic_Image/*.bmp'))
    path_to_masks = list(Path(dataset_root).rglob('*_lesion/*.bmp'))
    
    # apply filter_missing_images_and_masks and check the results    
    images, masks = filter_missing_images_and_masks(path_to_images, path_to_masks)
    assert images == expected_images, f"Paired images do not match expected paths: {images} vs {expected_images}"
    assert masks == expected_masks, f"Paired masks do not match expected paths: {masks} vs {expected_masks}"


