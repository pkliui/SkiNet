from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from SkiNet.ML.transformations.compute_dataset_stats import compute_stats, main


def _make_dataset(images: list) -> MagicMock:
    """Return a mock dataset whose get_raw_sample returns CHW uint8 arrays or tensors."""
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=len(images))

    def get_raw_sample(i: int) -> MagicMock:
        sample = MagicMock()
        sample.image = images[i]
        return sample

    dataset.get_raw_sample.side_effect = get_raw_sample
    return dataset


@pytest.fixture()
def patch_deps(tmp_path: Path) -> Generator[tuple[Path, MagicMock], None, None]:
    cfg_path = tmp_path / "dummy.yaml"
    cfg_path.touch()
    with (
        patch("SkiNet.ML.transformations.compute_dataset_stats.load_config_from_yaml"),
        patch("SkiNet.ML.transformations.compute_dataset_stats.create_segmentation_datasets_from_config") as mock_ds,
    ):
        yield cfg_path, mock_ds


def test_uniform_image_std_is_zero(patch_deps: tuple[Path, MagicMock]) -> None:
    """Identical constant images should give std == 0 for every channel."""
    cfg_path, mock_ds = patch_deps
    img = np.full((3, 4, 4), 128, dtype=np.uint8)
    mock_ds.return_value.train = _make_dataset([img, img, img])

    _, std = compute_stats(cfg_path)

    assert std == [0.0, 0.0, 0.0]


def test_uniform_image_mean_correct(patch_deps: tuple[Path, MagicMock]) -> None:
    """Constant 128 uint8 image should give mean == round(128/255, 4) per channel."""
    cfg_path, mock_ds = patch_deps
    img = np.full((3, 4, 4), 128, dtype=np.uint8)
    mock_ds.return_value.train = _make_dataset([img])

    mean, _ = compute_stats(cfg_path)

    expected = round(128 / 255, 4)
    assert mean == [expected, expected, expected]


def test_two_image_mean_and_std(patch_deps: tuple[Path, MagicMock]) -> None:
    """Test computing mean and std for a pair of all-zeros and all-255 images.

    Per-channel mean == 0.5, pixel-level std == 0.5.
    With mean = 0.5, the variance is (0-0.5)**2 + (1-0.5)**2 / 2 = 0.25, so std = 0.5."""
    cfg_path, mock_ds = patch_deps
    img0 = np.zeros((3, 2, 2), dtype=np.uint8)
    img1 = np.full((3, 2, 2), 255, dtype=np.uint8)
    mock_ds.return_value.train = _make_dataset([img0, img1])

    mean, std = compute_stats(cfg_path)

    assert mean == [0.5, 0.5, 0.5]
    assert std == [0.5, 0.5, 0.5]


def test_independent_channels(patch_deps: tuple[Path, MagicMock]) -> None:
    """Each channel has a distinct constant value; mean and std are computed independently."""
    cfg_path, mock_ds = patch_deps
    img = np.zeros((3, 4, 4), dtype=np.uint8)
    img[0] = 51
    img[1] = 128
    img[2] = 204
    mock_ds.return_value.train = _make_dataset([img])

    mean, std = compute_stats(cfg_path)

    assert mean[0] == round(51 / 255, 4)
    assert mean[1] == round(128 / 255, 4)
    assert mean[2] == round(204 / 255, 4)
    assert std == [0.0, 0.0, 0.0]


def test_accepts_torch_tensor_images(patch_deps: tuple[Path, MagicMock]) -> None:
    """compute_stats should handle torch.Tensor (CHW uint8) as well as numpy arrays."""
    cfg_path, mock_ds = patch_deps
    img = torch.full((3, 4, 4), 255, dtype=torch.uint8)
    mock_ds.return_value.train = _make_dataset([img])

    mean, std = compute_stats(cfg_path)

    assert mean == [1.0, 1.0, 1.0]
    assert std == [0.0, 0.0, 0.0]


def test_return_type_is_list_of_floats(patch_deps: tuple[Path, MagicMock]) -> None:
    """compute_stats must return two plain Python lists of floats, each of length 3."""
    cfg_path, mock_ds = patch_deps
    img = np.full((3, 8, 8), 100, dtype=np.uint8)
    mock_ds.return_value.train = _make_dataset([img])

    mean, std = compute_stats(cfg_path)

    assert isinstance(mean, list) and len(mean) == 3
    assert isinstance(std, list) and len(std) == 3
    assert all(isinstance(v, float) for v in mean + std)


def test_images_with_different_spatial_sizes(patch_deps: tuple[Path, MagicMock]) -> None:
    """
    Stats must be pixel-weighted, not image-weighted. This test uses three images whose
    sizes differ so that incorrect image-level counting produces a distinguishably wrong result.

    Images (per channel):
      img_a: 1×1, value 0   →  1 pixel, normalised 0.0
      img_b: 1×1, value 0   →  1 pixel, normalised 0.0
      img_c: 3×3, value 255 →  9 pixels, normalised 1.0

    Pixel-weighted mean:
      total_pixels = 1 + 1 + 9 = 11
      mean = (1*0 + 1*0 + 9*1) / 11 = 9/11 ≈ 0.8182

    Pixel-weighted std around pixel-weighted mean:
      sum_sq_diff = 1*(0 - 9/11)^2 + 1*(0 - 9/11)^2 + 9*(1 - 9/11)^2
                  = 2*(81/121) + 9*(4/121) = 162/121 + 36/121 = 198/121 = 18/11
      std = sqrt(18/11 / 11) = sqrt(18/121) = 3*sqrt(2)/11 ≈ 0.3857

    If images were treated as equal-weight (image-weighted mean = 1/3), mean ≈ 0.3333 — detectably wrong.
    """
    cfg_path, mock_ds = patch_deps
    img_a = np.zeros((3, 1, 1), dtype=np.uint8)
    img_b = np.zeros((3, 1, 1), dtype=np.uint8)
    img_c = np.full((3, 3, 3), 255, dtype=np.uint8)
    mock_ds.return_value.train = _make_dataset([img_a, img_b, img_c])

    mean, std = compute_stats(cfg_path)

    expected_mean = round(9 / 11, 4)
    expected_std = round(float(np.sqrt(18 / 121)), 4)
    assert mean == [expected_mean] * 3
    assert std == [expected_std] * 3


def test_main_prints_yaml_ready_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main() must print normalization_mean and normalization_std lines to stdout."""
    cfg_path = tmp_path / "dummy.yaml"
    cfg_path.touch()

    fixed_mean = [0.1234, 0.5678, 0.9012]
    fixed_std = [0.0111, 0.0222, 0.0333]

    with (
        patch("SkiNet.ML.transformations.compute_dataset_stats.compute_stats",
              return_value=(fixed_mean, fixed_std)),
        patch("sys.argv", ["compute_dataset_stats.py", "--config", str(cfg_path)]),
    ):
        main()

    captured = capsys.readouterr().out
    assert f"normalization_mean: {fixed_mean}" in captured
    assert f"normalization_std:  {fixed_std}" in captured
