import numpy as np
import pytest
import torch

from SkiNet.ML.transformations.transform_utils import convert_to_hwc_numpy

@pytest.mark.parametrize(
    "data_item,expected_shape",
    [
        (torch.ones((3, 8, 6), dtype=torch.uint8), (8, 6, 3)),
        (torch.ones((1, 8, 6), dtype=torch.uint8), (8, 6, 1)),
        (torch.ones((8, 6), dtype=torch.uint8), (8, 6, 1)),
        (np.ones((3, 8, 6), dtype=np.uint8), (8, 6, 3)),
        (np.ones((8, 6, 3), dtype=np.uint8), (8, 6, 3)),
        (np.ones((8, 6), dtype=np.uint8), (8, 6, 1)),
    ],
)
def test_convert_to_hwc_numpy_returns_expected_shapes(
    data_item: torch.Tensor | np.ndarray,
    expected_shape: tuple[int, int, int],
) -> None:
    """
    convert_to_hwc_numpy() should convert supported tensor and ndarray inputs into HWC-compatible numpy arrays.
    """
    result = convert_to_hwc_numpy(data_item)

    assert isinstance(result, np.ndarray)
    assert result.shape == expected_shape


@pytest.mark.parametrize(
    "data_item",
    [
        np.ones((2, 8, 6), dtype=np.uint8),
        np.ones((8, 6, 2), dtype=np.uint8),
        np.ones((8,), dtype=np.uint8),
        np.ones((2, 3, 4, 5), dtype=np.uint8),
    ],
)
def test_convert_to_hwc_numpy_raises_for_unsupported_shapes(data_item: np.ndarray) -> None:
    """
    convert_to_hwc_numpy() should raise ValueError for shapes that are not HW, CHW, or HWC with 1 or 3 channels.
    """
    with pytest.raises(ValueError):
        convert_to_hwc_numpy(data_item)
