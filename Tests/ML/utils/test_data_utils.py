import logging
import random
import tempfile
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pytest
from PIL import Image

from SkiNet.ML.utils.data_utils import (extract_sample_number, filter_images_and_masks_of_different_sizes,
                                        filter_missing_images_and_masks)

IMG_SIZE = 128
NUM_SAMPLES = 7

"""---------------------------------------------------------------------Testing extract_sample_number---------------------------------------------------"""

@pytest.mark.parametrize("filename, expected", [
    ("IMD001.bmp", 1),  # should return 1, not 001
    ("sample_42_mask.png", 42),  # should return 42
    ("test123.bmp", 123),  # should return 123
    ("no_number_here.bmp", 0),  # No digits should return 0
    ("random_99_test_1.tiff", 99),  # Takes first number found, should return 99
    ("100_sample11.tif", 100),  # 100 is at start, so should be extracted, not 11
    ("img.bmp", 0),  # No digits should return 0
])
def test_extract_sample_number(filename: str, expected: int) -> None:
    """
    Test that extract_sample_number correctly extracts numbers from filenames given assertions above
    """
    file_path = Path(filename)
    assert extract_sample_number(file_path) == expected


"""------------------------------Testing filter_missing_images_and_masks - local ph2 dataset-----------------------------"""


@pytest.fixture
def local_ph2dataset_one_sample_each_missing(tmp_path: Path) -> Any:
    """
    Create local_ph2dataset with missing image and mask samples
    """
    def _local_ph2dataset_one_sample_each_missing(image_sample_num_to_skip: List[int], mask_sample_num_to_skip: List[int]) -> Path:
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
                logging.getLogger(__name__).debug(f"Skipping image sample {ii} while creating local_ph2dataset for testing.")

            if ii not in mask_sample_num_to_skip:
                mask_folder = tmp_path / f"IMD{ii}" / f"IMD{ii}_lesion"
                mask_folder.mkdir(parents=True, exist_ok=True)
                mask = Image.fromarray(np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE), dtype=np.uint8))
                mask.save(mask_folder / f"IMD{ii}_lesion.bmp")
            else:
                logging.debug(f"Skipping mask sample {ii} while creating local_ph2dataset for testing.")
        return tmp_path

    yield _local_ph2dataset_one_sample_each_missing


@pytest.mark.parametrize(["image_sample_num_to_skip", "mask_sample_num_to_skip"],
                         [([2], [3])])
def test_pair_image_mask_paths_one_sample_missing(local_ph2dataset_one_sample_each_missing: Any,
                                                  image_sample_num_to_skip: List[int], mask_sample_num_to_skip: List[int]) -> None:
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
        dataset_root / "IMD0" / "IMD0_Dermoscopic_Image" / "IMD0.bmp",
        dataset_root / "IMD1" / "IMD1_Dermoscopic_Image" / "IMD1.bmp",
        dataset_root / "IMD4" / "IMD4_Dermoscopic_Image" / "IMD4.bmp",
        dataset_root / "IMD5" / "IMD5_Dermoscopic_Image" / "IMD5.bmp",
        dataset_root / "IMD6" / "IMD6_Dermoscopic_Image" / "IMD6.bmp"
    ]

    #  mask 3 missing
    expected_masks = [
        dataset_root / "IMD0" / "IMD0_lesion" / "IMD0_lesion.bmp",
        dataset_root / "IMD1" / "IMD1_lesion" / "IMD1_lesion.bmp",
        dataset_root / "IMD4" / "IMD4_lesion" / "IMD4_lesion.bmp",
        dataset_root / "IMD5" / "IMD5_lesion" / "IMD5_lesion.bmp",
        dataset_root / "IMD6" / "IMD6_lesion" / "IMD6_lesion.bmp"
    ]

    # get the actual paths to images and masks located in the root folder
    path_to_images = list(Path(dataset_root).rglob('*_Dermoscopic_Image/*.bmp'))
    path_to_masks = list(Path(dataset_root).rglob('*_lesion/*.bmp'))

    # apply filter_missing_images_and_masks and check the results
    images, masks = filter_missing_images_and_masks(path_to_images, path_to_masks)
    assert images == expected_images, f"Paired images do not match expected paths: {images} vs {expected_images}"
    assert masks == expected_masks, f"Paired masks do not match expected paths: {masks} vs {expected_masks}"


@pytest.fixture
def local_ph2dataset_one_sample_each_missing_unsorted(tmp_path: Path) -> Any:
    """
    Create local_ph2dataset with missing image and mask samples.
    Assume random order both for images and masks, i.e. samples cannot be paired (the first in the list of paths yiels image 0 and mask 1,
    for example
    """

    def _local_ph2dataset_one_sample_each_missing_unsorted(image_sample_num_to_skip: List[int],
                                                           mask_sample_num_to_skip: List[int]) -> Path:
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
def test_pair_image_mask_paths_one_sample_missing_unsorted(local_ph2dataset_one_sample_each_missing_unsorted: Any,
                                                           image_sample_num_to_skip: List[str], mask_sample_num_to_skip: List[str]) -> None:
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
        dataset_root / "IMD0" / "IMD0_Dermoscopic_Image" / "IMD0.bmp",
        dataset_root / "IMD1" / "IMD1_Dermoscopic_Image" / "IMD1.bmp",
        dataset_root / "IMD4" / "IMD4_Dermoscopic_Image" / "IMD4.bmp",
        dataset_root / "IMD5" / "IMD5_Dermoscopic_Image" / "IMD5.bmp",
        dataset_root / "IMD6" / "IMD6_Dermoscopic_Image" / "IMD6.bmp"
    ]

    #  mask 3 missing - we expect the list of paths to masks to be ordered after applying filter_missing_images_and_masks
    expected_masks = [
        dataset_root / "IMD0" / "IMD0_lesion" / "IMD0_lesion.bmp",
        dataset_root / "IMD1" / "IMD1_lesion" / "IMD1_lesion.bmp",
        dataset_root / "IMD4" / "IMD4_lesion" / "IMD4_lesion.bmp",
        dataset_root / "IMD5" / "IMD5_lesion" / "IMD5_lesion.bmp",
        dataset_root / "IMD6" / "IMD6_lesion" / "IMD6_lesion.bmp"
    ]

    # get the actual paths to images and masks located in the root folder
    path_to_images = list(Path(dataset_root).rglob('*_Dermoscopic_Image/*.bmp'))
    path_to_masks = list(Path(dataset_root).rglob('*_lesion/*.bmp'))

    # apply filter_missing_images_and_masks and check the results
    images, masks = filter_missing_images_and_masks(path_to_images, path_to_masks)
    assert images == expected_images, f"Paired images do not match expected paths: {images} vs {expected_images}"
    assert masks == expected_masks, f"Paired masks do not match expected paths: {masks} vs {expected_masks}"


"""------------------------------Testing filter_images_and_masks_of_different_sizes - arbitrary dataset------------------------------"""

def create_image(path: Path, size: Tuple[int, int] = (32, 32), color: int = 128) -> None:
    img = Image.new("L", size, color)
    img.save(path)

def test_filter_images_and_masks_of_different_sizes() -> None:
    """
    Test that filter_images_and_masks_of_different_sizes correctly filters out images and masks that do not have the same size.
    This test creates a temporary directory with images and masks of different sizes, then applies the filter function.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        # Create matching image/mask pairs
        img1 = tmp_dir / "img1.png"
        msk1 = tmp_dir / "msk1.png"
        create_image(img1, (32, 32))
        create_image(msk1, (32, 32))

        # Create mismatched image/mask pair
        img2 = tmp_dir / "img2.png"
        msk2 = tmp_dir / "msk2.png"
        create_image(img2, (32, 32))
        create_image(msk2, (34, 32))

        # Create mismatched image/mask pair
        img4 = tmp_dir / "img4.png"
        msk4 = tmp_dir / "msk4.png"
        create_image(img4, (32, 32))
        create_image(msk4, (33, 32))

        # Another matching pair
        img3 = tmp_dir / "img3.png"
        msk3 = tmp_dir / "msk3.png"
        create_image(img3, (16, 16))
        create_image(msk3, (16, 16))

        image_paths = [img1, img2, img3, img4]
        mask_paths = [msk1, msk2, msk3, msk4]

        filtered_imgs, filtered_msks = filter_images_and_masks_of_different_sizes(image_paths, mask_paths)

        # Only pairs (img1, msk1) and (img3, msk3) should remain
        assert filtered_imgs == [img1, img3]
        assert filtered_msks == [msk1, msk3]


"""-----------------------------------------Testing filter_and_pair_valid_pathss - arbitrary dataset - filter_if_size_different=True------------------------"""

from SkiNet.ML.utils.data_utils import filter_and_pair_valid_paths


def test_filter_and_pair_valid_paths_with_size_check() -> None:
    """
    Test that filter_and_pair_valid_paths returns only pairs that:
    - have both image and mask present (by sample number)

    - have the same size
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        # Valid pair
        img1 = tmp_dir / "img1.png"
        msk1 = tmp_dir / "img1_mask.png"
        create_image(img1, (32, 32))
        create_image(msk1, (32, 32))

        # Pair with missing mask
        img2 = tmp_dir / "img2.png"
        create_image(img2, (32, 32))
        # No mask2

        # Pair with missing image
        msk3 = tmp_dir / "img3_mask.png"
        create_image(msk3, (32, 32))
        # No img3

        # Pair with mismatched size
        img4 = tmp_dir / "img4.png"
        msk4 = tmp_dir / "img4_mask.png"
        create_image(img4, (32, 32))
        create_image(msk4, (16, 16))

        # Another valid pair
        img5 = tmp_dir / "img5.png"
        msk5 = tmp_dir / "img5_mask.png"
        create_image(img5, (24, 24))
        create_image(msk5, (24, 24))

        image_paths = [img1, img2, img4, img5]
        mask_paths = [msk1, msk3, msk4, msk5]

        filtered_imgs, filtered_msks = filter_and_pair_valid_paths(image_paths, mask_paths, True)

        # Only (img1, msk1) and (img5, msk5) should remain
        assert filtered_imgs == [img1, img5]
        assert filtered_msks == [msk1, msk5]


"""-------------------------------Testing filter_and_pair_valid_pathss - arbitrary dataset - filter_if_size_different=False----------------"""


def test_filter_and_pair_valid_paths_without_size_check() -> None:
    """
    Test that filter_and_pair_valid_paths returns only pairs that:
    - have both image and mask present (by sample number)
    - have the same size
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        # Valid pair
        img1 = tmp_dir / "img1.png"
        msk1 = tmp_dir / "img1_mask.png"
        create_image(img1, (32, 32))
        create_image(msk1, (32, 32))

        # Pair with missing mask
        img2 = tmp_dir / "img2.png"
        create_image(img2, (32, 32))
        # No mask2

        # Pair with missing image
        msk3 = tmp_dir / "img3_mask.png"
        create_image(msk3, (32, 32))
        # No img3

        # Pair with mismatched size
        img4 = tmp_dir / "img4.png"
        msk4 = tmp_dir / "img4_mask.png"
        create_image(img4, (32, 32))
        create_image(msk4, (16, 16))

        # Another valid pair
        img5 = tmp_dir / "img5.png"
        msk5 = tmp_dir / "img5_mask.png"
        create_image(img5, (24, 24))
        create_image(msk5, (24, 24))

        image_paths = [img1, img2, img4, img5]
        mask_paths = [msk1, msk3, msk4, msk5]

        filtered_imgs, filtered_msks = filter_and_pair_valid_paths(image_paths, mask_paths, False)

        # Only (img1, msk1) and (img5, msk5) should remain
        assert filtered_imgs == [img1, img4, img5]
        assert filtered_msks == [msk1, msk4, msk5]
