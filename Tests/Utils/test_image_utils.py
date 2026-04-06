import numpy as np
import pytest
import torch
from PIL import Image

from SkiNet.Utils.image_utils import ensure_np_image

IMG_HEIGHT = 64
IMG_WIDTH = 48
IMG_CHANNELS = 3


def test_ensure_np_image_numpy_passthrough_no_copy_required() -> None:
    arr = np.random.randint(
        0, 256, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8
    )
    out = ensure_np_image(arr)

    assert isinstance(out, np.ndarray)
    assert out.shape == (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    assert out.dtype == np.uint8
    assert out is arr


def test_ensure_np_image_pil_input_to_numpy_hwc_uint8() -> None:
    arr = np.random.randint(
        0, 256, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8
    )
    pil_img = Image.fromarray(arr)

    out = ensure_np_image(pil_img)

    assert isinstance(out, np.ndarray)
    assert out.shape == (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, arr)


def test_ensure_np_image_torch_hwc_tensor_to_numpy_hwc() -> None:
    tensor = torch.randint(
        0, 256, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=torch.uint8
    )

    out = ensure_np_image(tensor)

    assert isinstance(out, np.ndarray)
    assert out.shape == (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, tensor.numpy())


def test_ensure_np_image_torch_chw_transposes_to_hwc() -> None:
    tensor = torch.randint(
        0, 256, (IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH), dtype=torch.uint8
    )

    out = ensure_np_image(tensor)

    assert isinstance(out, np.ndarray)
    assert out.shape == (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    assert out.dtype == np.uint8

    expected = tensor.permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(out, expected)


def test_ensure_np_image_chw_with_single_channel_transposes_to_hwc() -> None:
    tensor = torch.randint(0, 256, (1, IMG_HEIGHT, IMG_WIDTH), dtype=torch.uint8)

    out = ensure_np_image(tensor)

    assert isinstance(out, np.ndarray)
    assert out.shape == (IMG_HEIGHT, IMG_WIDTH, 1)
    assert out.dtype == np.uint8

    expected = tensor.permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(out, expected)


def test_ensure_np_image_does_not_transpose_hwc_even_if_channels_equals_3() -> None:
    arr = np.random.randint(
        0, 256, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8
    )

    out = ensure_np_image(arr)

    assert out.shape == (IMG_HEIGHT, IMG_WIDTH, 3)
    np.testing.assert_array_equal(out, arr)


def test_ensure_np_image_bool_mask_converts_to_uint8() -> None:
    mask = np.random.rand(IMG_HEIGHT, IMG_WIDTH) > 0.5
    assert mask.dtype == bool

    out = ensure_np_image(mask)

    assert isinstance(out, np.ndarray)
    assert out.shape == (IMG_HEIGHT, IMG_WIDTH)
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 1})


def test_ensure_np_image_bool_chw_transpose_then_cast_to_uint8() -> None:
    mask = (np.random.rand(1, IMG_HEIGHT, IMG_WIDTH) > 0.5)
    assert mask.dtype == bool

    out = ensure_np_image(mask)

    assert isinstance(out, np.ndarray)
    assert out.shape == (IMG_HEIGHT, IMG_WIDTH, 1)
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 1})


@pytest.mark.parametrize(
    "shape",
    [
        (IMG_HEIGHT, IMG_WIDTH),          # 2D grayscale
        (IMG_HEIGHT, IMG_WIDTH, 4),       # HWC with alpha
        (4, IMG_HEIGHT, IMG_WIDTH),       # CHW but channels=4 should NOT transpose under current logic
    ],
)
def test_ensure_np_image_shapes_are_preserved_when_not_matching_transpose_rule(shape: tuple[int, ...]) -> None:
    arr = np.random.randint(0, 256, shape, dtype=np.uint8)

    out = ensure_np_image(arr)

    assert isinstance(out, np.ndarray)
    assert out.shape == shape
    assert out.dtype == np.uint8
