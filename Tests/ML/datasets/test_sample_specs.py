from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from SkiNet.ML.datasets.sample_specs import SampleSpecs, create_valid_samplespecs, load_data_item, load_sample
from SkiNet.Utils.csv_headers import DATAPATH_HEADER, DATATYPE_HEADER, DATATYPE_IMAGE, DATATYPE_MASK, SAMPLEID_HEADER


def _write_image(path: Path, array: np.ndarray) -> None:
    """
    Write a numpy array as an image to the specified path. The parent directory will be created if it does not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


@pytest.mark.parametrize(
    ("filename", "shape"),
    [
        ("img.png", (4, 4)),
        ("nested/mask.png", (3, 5)),
    ],
)
def test_load_data_item_returns_tensor(tmp_path: Path, filename: str, shape: tuple[int, int]) -> None:
    """
    Test that load_data_item returns a tensor with the expected shape and values
    """
    array = np.arange(shape[0] * shape[1], dtype=np.uint8).reshape(shape)
    _write_image(tmp_path / filename, array)

    tensor = load_data_item(filename, tmp_path)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == torch.Size(shape)
    assert torch.equal(tensor, torch.from_numpy(array))


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


def test_load_sample_loads_image_and_mask(tmp_path: Path) -> None:
    """
    Test that load_sample loads the image and mask tensors correctly.
    """
    image_array = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    mask_array = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    _write_image(tmp_path / "images/img.png", image_array)
    _write_image(tmp_path / "masks/mask.png", mask_array)

    specs = SampleSpecs(
        sample_id="sample-1",
        image_path="images/img.png",
        mask_path="masks/mask.png",
        metadata={"site": "A"},
    )

    sample = load_sample(specs, tmp_path)

    assert torch.equal(sample.image, torch.from_numpy(image_array))
    assert torch.equal(sample.mask, torch.from_numpy(mask_array))
    assert sample.specs == specs


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
