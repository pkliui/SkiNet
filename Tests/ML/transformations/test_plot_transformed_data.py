from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import pytest
import torch

from SkiNet.ML.datasets.sample_specs import Sample, SampleSpecs
from SkiNet.ML.transformations.plot_transformed_data import (
    _get_visualization_transform,
    _img_for_imsave,
    _img_for_overlay,
    _to_numpy_hwc_for_plot,
    _to_numpy_hw_for_plot,
    overlay_mask,
    visualize_augmented_data,
)
from SkiNet.ML.transformations.transform_adapters import AlbumentationsSampleTransform


class _FakeDataset:
    def __init__(self, transform: Any) -> None:
        self.transform = transform
        self.sample_ids = ["sample-1", "sample-2"]
        self.sample_specs = {
            "sample-1": SampleSpecs(sample_id="sample-1", image_path="image.png", mask_path="mask.png"),
            "sample-2": SampleSpecs(sample_id="sample-2", image_path="image2.png", mask_path="mask2.png"),
        }

    def __len__(self) -> int:
        return len(self.sample_ids)

    def get_raw_sample(self, idx: int) -> Sample:
        return Sample(
            image=torch.full((3, 32, 24), fill_value=idx, dtype=torch.uint8),
            mask=torch.ones((1, 32, 24), dtype=torch.uint8),
            specs=self.sample_specs[self.sample_ids[idx]],
        )


@pytest.mark.parametrize(
    "image,expected_shape",
    [
        (torch.ones((3, 8, 6), dtype=torch.uint8), (8, 6, 3)),
        (torch.ones((1, 8, 6), dtype=torch.uint8), (8, 6, 1)),
        (np.ones((8, 6, 3), dtype=np.uint8), (8, 6, 3)),
    ],
)
def test_to_numpy_hwc_for_plot_returns_expected_shapes(
    image: torch.Tensor | np.ndarray,
    expected_shape: tuple[int, int, int],
) -> None:
    """
    _to_numpy_hwc_for_plot() should return HWC numpy arrays for supported tensor and ndarray image inputs.
    """
    result = _to_numpy_hwc_for_plot(image)

    assert isinstance(result, np.ndarray)
    assert result.shape == expected_shape


def test_to_numpy_hwc_for_plot_raises_for_unsupported_type() -> None:
    """
    _to_numpy_hwc_for_plot() should raise TypeError for unsupported image input types.
    """
    with pytest.raises(TypeError, match="Unsupported image type for plotting"):
        _to_numpy_hwc_for_plot("not-an-image")


@pytest.mark.parametrize(
    "mask,expected_shape",
    [
        (torch.ones((1, 8, 6), dtype=torch.uint8), (8, 6)),
        (torch.ones((8, 6), dtype=torch.uint8), (8, 6)),
        (np.ones((8, 6, 1), dtype=np.uint8), (8, 6)),
        (np.ones((8, 6), dtype=np.uint8), (8, 6)),
    ],
)
def test_to_numpy_hw_for_plot_returns_expected_shapes(
    mask: torch.Tensor | np.ndarray,
    expected_shape: tuple[int, int],
) -> None:
    """
    _to_numpy_hw_for_plot() should return HW numpy arrays for supported tensor and ndarray mask inputs.
    """
    result = _to_numpy_hw_for_plot(mask)

    assert isinstance(result, np.ndarray)
    assert result.shape == expected_shape


def test_to_numpy_hw_for_plot_raises_for_unsupported_type() -> None:
    """
    _to_numpy_hw_for_plot() should raise TypeError for unsupported mask input types.
    """
    with pytest.raises(TypeError, match="Unsupported mask type for plotting"):
        _to_numpy_hw_for_plot("not-a-mask")


@pytest.mark.parametrize(
    "image,expected_dtype,expected_min,expected_max",
    [
        (np.array([[0, 255]], dtype=np.uint8), np.uint8, 0.0, 255.0),
        (np.array([[0.25, 0.75]], dtype=np.float32), np.float32, 0.25, 0.75),
        (np.array([[2.0, 6.0]], dtype=np.float32), np.float32, 0.0, 1.0),
        (np.array([[300, -10]], dtype=np.int32), np.uint8, 0.0, 255.0),
    ],
)
def test_img_for_imsave_normalizes_or_clips_values(
    image: np.ndarray,
    expected_dtype: np.dtype,
    expected_min: float,
    expected_max: float,
) -> None:
    """
    _img_for_imsave() should preserve display-safe inputs and normalize or clip out-of-range inputs.
    """
    result = _img_for_imsave(image)

    assert result.dtype == expected_dtype
    assert float(np.nanmin(result)) == expected_min
    assert float(np.nanmax(result)) == expected_max


@pytest.mark.parametrize(
    "image,expected_dtype,expected_min,expected_max",
    [
        (np.array([[0, 255]], dtype=np.uint8), np.uint8, 0.0, 255.0),
        (np.array([[0.0, 1.0]], dtype=np.float32), np.uint8, 0.0, 255.0),
    ],
)
def test_img_for_overlay_returns_uint8_image_range(
    image: np.ndarray,
    expected_dtype: np.dtype,
    expected_min: float,
    expected_max: float,
) -> None:
    """
    _img_for_overlay() should always produce uint8 image data in the 0..255 range.
    """
    result = _img_for_overlay(image)

    assert result.dtype == expected_dtype
    assert float(result.min()) == expected_min
    assert float(result.max()) == expected_max


def test_overlay_mask_returns_colorized_image() -> None:
    """
    overlay_mask() should blend the configured mask color into positive mask pixels.
    """
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1

    result = overlay_mask(image, mask, alpha=0.5, color=(0, 1, 0))

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.all(result[0, 0] == np.array([0, 0, 0], dtype=np.uint8))
    assert result[1, 1, 1] > 0


@pytest.mark.parametrize("dataset_transform", [None, lambda sample: sample])
def test_get_visualization_transform_returns_none_or_passthrough(dataset_transform: Any) -> None:
    """
    _get_visualization_transform() should return None unchanged and preserve plain callable transforms.
    """
    result = _get_visualization_transform(dataset_transform)

    assert result is dataset_transform


def test_get_visualization_transform_uses_without_postprocess_when_available() -> None:
    """
    _get_visualization_transform() should call without_postprocess() when the dataset transform exposes it.
    """
    dataset_transform = AlbumentationsSampleTransform(
        pipeline=A.Compose([A.CenterCrop(height=16, width=12), A.ToTensorV2(transpose_mask=True)]),
        visualization_pipeline=A.Compose([A.CenterCrop(height=16, width=12)]),
        expects_tensor_output=True,
    )

    result = _get_visualization_transform(dataset_transform)

    assert isinstance(result, AlbumentationsSampleTransform)
    assert result.expects_tensor_output is False


def test_visualize_augmented_data_saves_expected_files(tmp_path: Path) -> None:
    """
    visualize_augmented_data() saves per-sample overlays and a single grid file.
    Each randomly chosen sample produces: orig_overlay, aug_overlay, aug_mask_not_binary.
    """
    dataset = _FakeDataset(
        AlbumentationsSampleTransform(
            pipeline=A.Compose([A.CenterCrop(height=16, width=12), A.ToTensorV2(transpose_mask=True)]),
            visualization_pipeline=A.Compose([A.CenterCrop(height=16, width=12)]),
            expects_tensor_output=True,
        )
    )

    visualize_augmented_data(
        dataset=dataset,
        samples=2,
        save_dir=tmp_path,
        prefix="vis",
        show=False,
    )

    # one grid file always saved
    assert (tmp_path / "vis_grid.png").exists()

    # per-sample files: 2 samples * 3 files each = 6
    per_sample_files = [f for f in tmp_path.glob("vis_*.png") if "grid" not in f.name]
    assert len(per_sample_files) == 6

    suffixes = {f.name.split("idx")[1].split("_", 1)[1] for f in per_sample_files}
    assert "orig_overlay.png" in suffixes
    assert "aug_overlay.png" in suffixes
    assert "aug_mask_not_binary.png" in suffixes


def test_visualize_augmented_data_restores_rng_state(tmp_path: Path) -> None:
    """RNG state after visualize_augmented_data() must be identical to state before."""
    dataset = _FakeDataset(transform=None)

    torch.manual_seed(42)
    state_before = torch.get_rng_state()

    visualize_augmented_data(dataset=dataset, samples=2, save_dir=None, show=False)

    state_after = torch.get_rng_state()
    assert torch.equal(state_before, state_after)
