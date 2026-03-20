import numpy as np
import pytest
import torch

from SkiNet.ML.datasets.sample_specs import Sample, SampleSpecs
from SkiNet.ML.transformations.crop_data import crop_2d_image


def make_sample(image: np.ndarray) -> Sample:
    # Convert numpy array to torch.Tensor for type compatibility
    specs = SampleSpecs(sample_id="dummy", image_path="", mask_path="", metadata={})
    torch_image = torch.from_numpy(image)
    torch_mask = torch.zeros_like(torch_image)  # Dummy mask, same shape as image
    return Sample(image=torch_image, mask=torch_mask, specs=specs)


@pytest.mark.parametrize("img_shape,crop_size,expected_shape", [
    ((112, 110), (100, 100), (100, 100)),
    ((572, 463), (460, 460), (460, 460)),
    ((571, 462), (460, 460), (460, 460)),
    ((570, 461), (460, 460), (460, 460)),
    ((572, 460), (460, 460), (460, 460)),
])
def test_crop_2d_image_shapes(img_shape: tuple[int, int], crop_size: tuple[int, int], expected_shape: tuple[int, int]) -> None:
    img = np.arange(np.prod(img_shape)).reshape(img_shape)
    sample = make_sample(img)
    slices = crop_2d_image(sample, crop_size)
    cropped = img[slices[0], slices[1]]
    assert cropped.shape == expected_shape

@pytest.mark.parametrize("img_shape,crop_size", [
    ((572, 460), (470, 470)),
    ((100, 100), (200, 200)),
    ((50, 50), (50, 51)),
])
def test_crop_2d_image_shapes_error(img_shape: tuple[int, int], crop_size: tuple[int, int]) -> None:
    img = np.arange(np.prod(img_shape)).reshape(img_shape)
    sample = make_sample(img)
    with pytest.raises(ValueError, match="exceeds image size"):
        crop_2d_image(sample, crop_size)
