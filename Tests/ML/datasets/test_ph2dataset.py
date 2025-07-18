"""Tests for SkiNet.ML.datasets.ph2dataset.py"""


# linear size of image or mask
IMG_HEIGHT = 576
IMG_WIDTH =768
IMG_CHANNELS = 3

# number of sample pairs for testing
NUM_SAMPLES = 5

AZURE_NUM_SAMPLES = 3

"""------------------------------------------------------------------FIXTURES---------------------------------------------------------------"""
from PIL import Image
import numpy as np
import pytest
import torch

from SkiNet.Utils import project_paths

@pytest.fixture
def local_ph2dataset(tmp_path):
    """
    Create a temporary PH2-like dataset locally with valid image/mask pairs, i.e. where for each sample number there are both an image and a mask
    """
    for ii in range (0, NUM_SAMPLES):
        # Create image folder & mask folder
        img_folder = tmp_path / f"IMD{ii}" / f"IMD{ii}_Dermoscopic_Image"
        mask_folder = tmp_path / f"IMD{ii}" / f"IMD{ii}_lesion"
        img_folder.mkdir(parents=True, exist_ok=True)
        mask_folder.mkdir(parents=True, exist_ok=True)

        # Create image and mask
        img = Image.fromarray(np.random.randint(0, 255, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8))
        mask = Image.fromarray(np.random.randint(0, 255, (IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8))
        img.save(img_folder / f"IMD{ii}.bmp")
        mask.save(mask_folder / f"IMD{ii}_lesion.bmp") 

    yield tmp_path

"""------------------------------------------------------------------TESTS for get_ph2_data_paths - exceptions---------------------------------------------------------------"""

from SkiNet.ML.datasets.ph2dataset import PH2Dataset, get_ph2_data_paths

def test_get_ph2_data_paths_local_path_does_not_exist():
    """
    Test that get_ph2_data_paths raises Value error if local path does not exist
    """
    with pytest.raises(ValueError):
        _,_ = get_ph2_data_paths(data_root="it_does_not_exist", azure_data=False)
    
def test_get_ph2_data_paths_local_path_invalid_type():
    """
    Test that get_ph2_data_paths raises Value error if local paths is not of a valid type
    """
    invalid_inputs = [1, 1.0, True, [], (), {}, None]
    for input_value in invalid_inputs:
        with pytest.raises(ValueError):
            _,_ = get_ph2_data_paths(data_root=input_value, azure_data=False)

"""------------------------------------------------------------------TESTS for get_ph2_data_paths - local data---------------------------------------------------------------"""

from pathlib import Path

def test_get_ph2_data_paths_local_ph2dataset(local_ph2dataset):
    """
    Test that get_ph2_data_paths correctly finds and returns the file paths
    for images and masks from the temporary local dataset structure
    """

    dataset_root = local_ph2dataset

    # Call the function under test with azure_data set to False.
    image_paths, mask_paths = get_ph2_data_paths(data_root=dataset_root, azure_data=False)
    image_paths = [Path(p.decode("utf-8")) if isinstance(p, (bytes, np.bytes_)) else p for p in image_paths]
    mask_paths = [Path(p.decode("utf-8"))if isinstance(p, (bytes, np.bytes_)) else p for p in mask_paths]
    
    # Validate the number of image and mask paths returned
    assert len(image_paths) == NUM_SAMPLES
    assert len(mask_paths) == NUM_SAMPLES

    # Check that each file exists and of correct size
    for img_path in image_paths:
        assert Path(img_path).exists()
        with Image.open(img_path) as img:
            assert img.size == (IMG_WIDTH, IMG_HEIGHT)
            assert img.mode == "RGB"
            # check after converting to numpy array
            img_array = np.array(img)
            assert img_array.shape == (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
            
    for mask_path in mask_paths:
        assert Path(mask_path).exists()
        with Image.open(mask_path) as mask:
            assert mask.size == (IMG_WIDTH, IMG_HEIGHT)
            assert mask.mode == "L"
            # check after converting to numpy array
            mask_array = np.array(mask)
            assert mask_array.shape == (IMG_HEIGHT, IMG_WIDTH)

    # Verify paths to images and masks
    expected_images = [
        dataset_root / f"IMD0" / f"IMD0_Dermoscopic_Image" / f"IMD0.bmp",
        dataset_root / f"IMD1" / f"IMD1_Dermoscopic_Image" / f"IMD1.bmp",
        dataset_root / f"IMD2" / f"IMD2_Dermoscopic_Image" / f"IMD2.bmp",
        dataset_root / f"IMD3" / f"IMD3_Dermoscopic_Image" / f"IMD3.bmp",
        dataset_root / f"IMD4" / f"IMD4_Dermoscopic_Image" / f"IMD4.bmp",
    ]
    expected_masks = [
        dataset_root / f"IMD0" / f"IMD0_lesion" / f"IMD0_lesion.bmp",
        dataset_root / f"IMD1" / f"IMD1_lesion" / f"IMD1_lesion.bmp",
        dataset_root / f"IMD2" / f"IMD2_lesion" / f"IMD2_lesion.bmp",
        dataset_root / f"IMD3" / f"IMD3_lesion" / f"IMD3_lesion.bmp",
        dataset_root / f"IMD4" / f"IMD4_lesion" / f"IMD4_lesion.bmp",
    ]
    

    def _to_path(p):
        # Decode if np.bytes_ or bytes, then convert to Path
        if isinstance(p, (bytes, np.bytes_)):
            return Path(p.decode("utf-8"))
        return Path(p)

    # Sort both lists by the extracted sample number - glob.glob has no guarantee to return paths in any order
    sort_key = lambda p: int(''.join(filter(str.isdigit, _to_path(p).stem)) or 0)
    assert sorted(image_paths, key=sort_key) == expected_images, f"Image paths do not match expected: {image_paths} vs {expected_images}"
    assert sorted(mask_paths, key=sort_key) == expected_masks, f"Mask paths do not match expected: {mask_paths} vs {expected_masks}"


"""------------------------------------------------------------------TESTS for ph2dataset - local dataset ---------------------------------------------------------------"""

def test_ph2dataset_len_getitem_no_transform(local_ph2dataset):
    """
    Test the PH2Dataset class:
        - __len__ should return NUM_SAMPLES.
        - __getitem__ should return a dict with keys "image" and "mask" that are torch tensors of shape 3 and 2 for images and masks
    """
    # Instantiate PH2Dataset with filtering enabled (default)
    #dataset = PH2Dataset(data_root=str("workplace/SkiNet/PH2_Dataset_images"))
    dataset = PH2Dataset(data_root=local_ph2dataset)

    # Check dataset length
    assert len(dataset) == NUM_SAMPLES, f"Expected dataset length {NUM_SAMPLES}, got {len(dataset)}."

    # Test __getitem__ for each index
    for idx in range(len(dataset)):
        sample = dataset[idx]
        assert isinstance(sample, dict), f"Sample at index {idx} is not a dict."
        assert "image" in sample, f"Sample at index {idx} missing 'image'."
        assert "mask" in sample, f"Sample at index {idx} missing 'mask'."
        # Verify that images and masks are loaded as torch tensors
        assert isinstance(sample["image"], torch.Tensor), f"Image at index {idx} is not a torch.Tensor."
        assert isinstance(sample["mask"], torch.Tensor), f"Mask at index {idx} is not a torch.Tensor."

        assert sample["image"].shape[0] == 3  # Channels
        assert sample["image"].ndim == 3
        assert sample["mask"].ndim == 2


import albumentations as A

def test_ph2dataset_getitem_with_custom_transform(local_ph2dataset):
    """
    Test that the PH2Dataset dataclass returns a dict with keys "image" and "mask" as per provided custom transform
    """
    # Albumentations transform
    transform = A.Compose([
        A.Resize(height=512, width=512),
        A.ToTensorV2()
    ])
    dataset = PH2Dataset(data_root=local_ph2dataset, transform=transform)
    sample = dataset[0]
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["mask"], torch.Tensor)
    assert sample["image"].shape == (IMG_CHANNELS, 512, 512)
    assert sample["mask"].shape == (512, 512)


def test_ph2dataset_default_visualisation_transform(local_ph2dataset):
    """
    Test that the PH2Dataset dataclass returns a dict with keys "image" and "mask" as per default transform
    """
    dataset = PH2Dataset(data_root=local_ph2dataset, default_transform_visualisation=True)
    sample = dataset[0]
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape== (IMG_CHANNELS, 510, 510)
    assert sample["mask"].shape == (510, 510)





"""------------------------------------------------------------------ INTEGRATION TESTS for ph2dataset - Azure dataset ---------------------------------------------------------------"""

@pytest.mark.azure
def test_azure_ph2dataset_len_getitem_no_transform():

    """
    Test getting data from Azure

    Check if the number of samples is correct,
    check dataset returns a dictionary with keys "image" and "mask",
    check they are tensors and their overall dimensions
    """

    from SkiNet.Azure.azure_setup import AzureSetup
    AzureSetup.service_principal_authentication()

    fs = AzureSetup.get_azureml_filesystem("PH2")
    dataset = PH2Dataset(data_root=fs)

    # Check dataset length
    assert len(dataset) == AZURE_NUM_SAMPLES, f"Expected dataset length {AZURE_NUM_SAMPLES}, got {len(dataset)}."

    # Test __getitem__ for each index
    for idx in range(len(dataset)):
        sample = dataset[idx]
        assert isinstance(sample, dict), f"Sample at index {idx} is not a dict."
        assert "image" in sample, f"Sample at index {idx} missing 'image'."
        assert "mask" in sample, f"Sample at index {idx} missing 'mask'."
        # Verify that images and masks are loaded as torch tensors
        assert isinstance(sample["image"], torch.Tensor), f"Image at index {idx} is not a torch.Tensor."
        assert isinstance(sample["mask"], torch.Tensor), f"Mask at index {idx} is not a torch.Tensor."
        assert sample["image"].shape[0] == 3  # Channels
        assert sample["image"].ndim == 3
        assert sample["mask"].ndim == 2

