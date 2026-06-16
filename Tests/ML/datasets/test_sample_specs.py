from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from SkiNet.ML.datasets.sample_specs import SampleSpecs, create_valid_samplespecs, load_data_item, load_sample
from SkiNet.Utils.csv_headers import DATAPATH_HEADER, DATATYPE_HEADER, DATATYPE_IMAGE, DATATYPE_MASK, SAMPLEID_HEADER


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


@pytest.mark.parametrize(
    "img_rel,mask_rel,img_array,mask_array,expected_img_shape,expected_mask_shape",
    [
        (
            Path("images/img_gray.png"),
            Path("masks/mask_gray.png"),
            np.random.randint(0, 256, (8, 6), dtype=np.uint8),  # grayscale (H, W)
            np.random.randint(0, 256, (8, 6), dtype=np.uint8),  # grayscale mask (H, W)
            (1, 8, 6),  # decode_image returns (C, H, W) where C=1 for grayscale images
            (1, 8, 6),
        ),
        (
            Path("images/img_rgb.png"),
            Path("masks/mask_gray.png"),
            np.random.randint(0, 256, (7, 9, 3), dtype=np.uint8),  # RGB (H, W, C)
            np.random.randint(0, 256, (7, 9), dtype=np.uint8),  # grayscale mask (H, W)
            (3, 7, 9),  # decode_image returns (C, H, W) where C=3 for RGB images
            (1, 7, 9),
        ),
        (
            Path("nested/images/img_gray.png"),
            Path("nested/masks/mask_gray.png"),
            np.random.randint(0, 256, (4, 5), dtype=np.uint8),
            np.random.randint(0, 256, (4, 5), dtype=np.uint8),
            (1, 4, 5),  # decode_image returns (C, H, W) where C=1 for grayscale images
            (1, 4, 5),
        ),
    ],
)
def test_load_sample_reads_image_and_mask_and_preserves_specs(
    tmp_path: Path,
    img_rel: Path,
    mask_rel: Path,
    img_array: np.ndarray,
    mask_array: np.ndarray,
    expected_img_shape: tuple[int, int, int],
    expected_mask_shape: tuple[int, int, int],
) -> None:
    data_root = tmp_path
    _write_png(data_root / img_rel, img_array)
    _write_png(data_root / mask_rel, mask_array)

    specs = SampleSpecs(sample_id="sample-1", image_path=str(img_rel), mask_path=str(mask_rel))

    sample = load_sample(specs, data_root=data_root)

    assert isinstance(sample.image, torch.Tensor)
    assert isinstance(sample.mask, torch.Tensor)
    assert sample.image.dtype == torch.uint8
    assert sample.mask.dtype == torch.uint8
    assert tuple(sample.image.shape) == expected_img_shape
    assert tuple(sample.mask.shape) == expected_mask_shape
    assert sample.specs == specs


@pytest.mark.parametrize(
    "rel_path,array,expected_shape",
    [
        (Path("img_gray.png"), np.random.randint(0, 256, (8, 6), dtype=np.uint8), (1, 8, 6)),
        (Path("img_rgb.png"), np.random.randint(0, 256, (7, 9, 3), dtype=np.uint8), (3, 7, 9)),
        (Path("nested/mask_gray.png"), np.random.randint(0, 256, (4, 5), dtype=np.uint8), (1, 4, 5)),
    ],
)
def test_load_data_item_returns_chw_uint8(
    tmp_path: Path,
    rel_path: Path,
    array: np.ndarray,
    expected_shape: tuple[int, int, int],
) -> None:
    _write_png(tmp_path / rel_path, array)

    out = load_data_item(str(rel_path), tmp_path)

    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.uint8
    assert out.ndim == 3
    assert tuple(out.shape) == expected_shape


def test_load_data_item_raises_when_data_root_missing(tmp_path: Path) -> None:
    """
    Test that load_data_item raises a FileNotFoundError when the data root is missing.
    """
    missing_root = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError, match="Data root does not exist"):
        load_data_item("img.png", missing_root)


@pytest.mark.parametrize(
    "filename",
    [
        "missing.png",
        "nested/missing_mask.png",
    ],
)
def test_load_data_item_raises_when_file_missing(tmp_path: Path, filename: str) -> None:
    """
    Test that load_data_item raises a FileNotFoundError when the file is missing.
    """
    with pytest.raises(FileNotFoundError, match="Data item not found"):
        load_data_item(filename, tmp_path)


def test_create_valid_samplespecs_returns_expected_sample(tmp_path: Path) -> None:
    """
    Test that create_valid_samplespecs returns the expected SampleSpecs object.
    """
    df = pd.DataFrame(
        [
            {
                SAMPLEID_HEADER: "sample-1",
                DATATYPE_HEADER: DATATYPE_IMAGE,
                DATAPATH_HEADER: "images/img1.png",
                "site": "A",
            },
            {
                SAMPLEID_HEADER: "sample-1",
                DATATYPE_HEADER: DATATYPE_MASK,
                DATAPATH_HEADER: "masks/mask1.png",
                "site": "A",
            },
        ]
    )

    result = create_valid_samplespecs(df)

    assert list(result.keys()) == ["sample-1"]
    specs = result["sample-1"]
    assert specs.sample_id == "sample-1"
    assert specs.image_path == "images/img1.png"
    assert specs.mask_path == "masks/mask1.png"
    assert specs.metadata == {"site": "A"}


def r(sample_id: str, dtype: str, path: str, site: str) -> dict:
    return {
        SAMPLEID_HEADER: sample_id,
        DATATYPE_HEADER: dtype,
        DATAPATH_HEADER: path,
        "site": site,
    }


@pytest.mark.parametrize(
    "rows, expected_keys",
    [
        ([r("sample-1", DATATYPE_IMAGE, "images/img1.png", "A")], []),
        ([r("sample-1", DATATYPE_MASK, "masks/mask1.png", "A")], []),
        (
            [r("sample-1", DATATYPE_IMAGE, "images/img1.png", "A"),
             r("sample-1", DATATYPE_MASK, "masks/mask1.png", "B")],
            [],
        ),
        (
            [r("sample-1", DATATYPE_IMAGE, "images/img1.png", "A"),
             r("sample-1", DATATYPE_MASK, "masks/mask1.png", "A"),
             r("sample-2", DATATYPE_IMAGE, "images/img2.png", "B"),
             r("sample-2", DATATYPE_MASK, "masks/mask2.png", "B")],
            ["sample-1", "sample-2"],
        ),
    ],
)
def test_create_valid_samplespecs_parameterized(rows: list[dict], expected_keys: list[str]) -> None:
    """
    Test that create_valid_samplespecs returns the expected keys for the given input rows.
    """
    df = pd.DataFrame(rows)

    result = create_valid_samplespecs(df)

    assert list(result.keys()) == expected_keys


def test_create_valid_samplespecs_uses_first_image_and_mask_when_duplicates(caplog: pytest.LogCaptureFixture) -> None:
    """
    Test that create_valid_samplespecs uses the first image and mask when duplicates are present.
    """
    df = pd.DataFrame(
        [
            {
                SAMPLEID_HEADER: "sample-1",
                DATATYPE_HEADER: DATATYPE_IMAGE,
                DATAPATH_HEADER: "images/img1.png",
                "site": "A",
            },
            {
                SAMPLEID_HEADER: "sample-1",
                DATATYPE_HEADER: DATATYPE_IMAGE,
                DATAPATH_HEADER: "images/img1b.png",
                "site": "A",
            },
            {
                SAMPLEID_HEADER: "sample-1",
                DATATYPE_HEADER: DATATYPE_MASK,
                DATAPATH_HEADER: "masks/mask1.png",
                "site": "A",
            },
            {
                SAMPLEID_HEADER: "sample-1",
                DATATYPE_HEADER: DATATYPE_MASK,
                DATAPATH_HEADER: "masks/mask1b.png",
                "site": "A",
            },
        ]
    )

    with caplog.at_level("WARNING"):
        result = create_valid_samplespecs(df)

    specs = result["sample-1"]
    assert specs.image_path == "images/img1.png"
    assert specs.mask_path == "masks/mask1.png"
    assert "Multiple images found for sample_id sample-1" in caplog.text
    assert "Multiple masks found for sample_id sample-1" in caplog.text
