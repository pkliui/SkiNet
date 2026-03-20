import numpy as np
import pytest
import torch

from SkiNet.ML.datasets.sample_specs import Sample, SampleSpecs
from SkiNet.ML.transformations.crop_data import crop_2d_image


def make_sample(image: np.ndarray, mask: np.ndarray) -> Sample:
    """
    Create a Sample object from image and mask numpy arrays.
    """
    specs = SampleSpecs(sample_id="dummy", image_path="", mask_path="", metadata={})
    torch_image = torch.from_numpy(image)
    torch_mask = torch.from_numpy(mask)
    return Sample(image=torch_image, mask=torch_mask, specs=specs)

@pytest.mark.parametrize("img_shape,crop_size,expected_img_shape", [
    ((1, 572, 765, 3), (256, 256), (1, 256, 256, 3)),
    ((1, 460, 460, 3), (200, 200), (1, 200, 200, 3)),
    ((1, 462, 461, 3), (460, 460), (1, 460, 460, 3)),
])
def test_crop_2d_image_shapes_4d(img_shape: tuple[int, int, int, int], crop_size: tuple[int, int], expected_img_shape: tuple[int, int, int, int]) -> None:
    """
    Test cropping of 4D image tensors.
    """
    img = np.random.rand(*img_shape).astype(np.float32)
    mask = np.random.randint(0, 2, size=(img_shape[0], img_shape[1], img_shape[2])).astype(np.int64)
    sample = make_sample(img, mask)
    slices = crop_2d_image(sample.image, crop_size)
    cropped_img = sample.image[slices]
    assert cropped_img.shape == expected_img_shape

@pytest.mark.parametrize("mask_shape,crop_size,expected_mask_shape", [
    ((1, 572, 765), (256, 256), (1, 256, 256)),
    ((1, 460, 460), (200, 200), (1, 200, 200)),
    ((1, 462, 461), (460, 460), (1, 460, 460)),
])
def test_crop_2d_image_shapes_3d(mask_shape: tuple[int, int, int], crop_size: tuple[int, int], expected_mask_shape: tuple[int, int, int]) -> None:
    """
    Test cropping of 3D mask tensors.
    """
    img = np.random.rand(mask_shape[0], mask_shape[1], mask_shape[2], 3).astype(np.float32)
    mask = np.random.randint(0, 2, size=mask_shape).astype(np.int64)
    sample = make_sample(img, mask)
    slices = crop_2d_image(sample.mask, crop_size)
    cropped_mask = sample.mask[slices]
    assert cropped_mask.shape == expected_mask_shape

@pytest.mark.parametrize("img_shape,crop_size", [
    ((1, 572, 765, 3), (600, 800)),
    ((1, 100, 100, 3), (200, 200)),
])
def test_crop_2d_image_shapes_error_4d(img_shape: tuple[int, int, int, int], crop_size: tuple[int, int]) -> None:
    """
    Test cropping of 4D image tensors with invalid crop sizes.
    """
    img = np.random.rand(*img_shape).astype(np.float32)
    mask = np.random.randint(0, 2, size=(img_shape[0], img_shape[1], img_shape[2])).astype(np.int64)
    sample = make_sample(img, mask)
    with pytest.raises(ValueError, match="exceeds image size"):
        crop_2d_image(sample.image, crop_size)

@pytest.mark.parametrize("mask_shape,crop_size", [
    ((1, 572, 765), (600, 800)),
    ((1, 100, 100), (200, 200)),
])
def test_crop_2d_image_shapes_error_3d(mask_shape: tuple[int, int, int], crop_size: tuple[int, int]) -> None:
    """
    Test cropping of 3D mask tensors with invalid crop sizes.
    """
    img = np.random.rand(mask_shape[0], mask_shape[1], mask_shape[2], 3).astype(np.float32)
    mask = np.random.randint(0, 2, size=mask_shape).astype(np.int64)
    sample = make_sample(img, mask)
    with pytest.raises(ValueError, match="exceeds image size"):
        crop_2d_image(sample.mask, crop_size)
