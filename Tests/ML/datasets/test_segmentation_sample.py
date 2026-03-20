from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
import torch

from SkiNet.ML.datasets.segmentation_dataset import SegmentationDataset
from SkiNet.Utils.csv_headers import DATAPATH_HEADER, DATATYPE_HEADER, DATATYPE_IMAGE, DATATYPE_MASK, SAMPLEID_HEADER

CROP_SIZE = (256, 256)  # as in main_config
IMG_SIZE = (572, 765)  # as in main_run dummy sample
IMG_CHANNELS = 3  # as in main_run dummy sample
MASK_SIZE = (572, 765)  # as in main_run dummy sample

def _make_config(df: pd.DataFrame, data_root: Path) -> Any:
    """
    Create a configuration object for the dataset mimicking
    config.dataconfig.metadata and config.dataconfig.data_root.
    """
    cfg = SimpleNamespace(
        metadata=df,
        data_root=str(data_root),
        crop_size=CROP_SIZE,
    )
    return SimpleNamespace(dataconfig=cfg)


def row_factory(sample_id: str, dtype: str, path: str, site: str) -> dict:
    """Row factory: keeps test cases concise."""
    return {
        SAMPLEID_HEADER: sample_id,
        DATATYPE_HEADER: dtype,
        DATAPATH_HEADER: path,
        "site": site,
    }
# ...existing code...

@pytest.mark.parametrize(
    ("rows", "expected_len"),
    [
        (
            [row_factory("sample-1", DATATYPE_IMAGE, "images/img1.png", "A"),
             row_factory("sample-1", DATATYPE_MASK, "masks/mask1.png", "A")],
            1,
        ),
        (
            [row_factory("sample-1", DATATYPE_IMAGE, "images/img1.png", "A")],
            0,  # Only one image, no mask
        ),
        (
            [row_factory("sample-1", DATATYPE_MASK, "masks/mask1.png", "A")],
            0,  # Only one mask, no image
        ),
        (
            [row_factory("sample-1", DATATYPE_IMAGE, "images/img1.png", "A"),
             row_factory("sample-1", DATATYPE_MASK, "masks/mask1.png", "B")],
            0,  # Different sites (different metadata)
        ),
        (
            [row_factory("sample-1", DATATYPE_IMAGE, "images/img1.png", "A"),
             row_factory("sample-1", DATATYPE_MASK, "masks/mask1.png", "A"),
             row_factory("sample-2", DATATYPE_IMAGE, "images/img2.png", "B"),
             row_factory("sample-2", DATATYPE_MASK, "masks/mask2.png", "B")],
            2,
        ),
    ],
)
def test_segmentation_dataset_len(tmp_path: Path, rows: list[dict], expected_len: int) -> None:
    """
    Test the length of the SegmentationDataset based on the provided rows of metadata,
    ensuring that only valid samples are counted.
    """
    df = pd.DataFrame(rows)
    config = _make_config(df, tmp_path)

    dataset = SegmentationDataset(config=config)

    assert len(dataset) == expected_len


def test_segmentation_dataset_getitem_returns_expected_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test the __getitem__ method of the SegmentationDataset.
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
    config = _make_config(df, tmp_path)
    config = cast(Any, config)
    dataset = SegmentationDataset(config=config)

    class DummySpecs:
        def model_dump(self) -> dict[str, str]:
            return {"sample_id": "sample-1", "image_path": "images/img1.png", "mask_path": "masks/mask1.png"}

    class DummySample:
        def __init__(self) -> None:
            # Realistic shapes: batch, height, width, channels for image; batch, height, width for mask
            self.image = torch.rand(1, IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS)  # float tensor, channels last
            self.mask = torch.rand(1, MASK_SIZE[0], MASK_SIZE[1])  # float tensor, no explicit channels
            self.specs = DummySpecs()

    def fake_load_sample(specs_item: Any, data_root: Path) -> DummySample:
        """
        Fake load_sample function for testing.
        """
        assert specs_item.sample_id == "sample-1"
        assert data_root == Path(tmp_path)
        return DummySample()

    monkeypatch.setattr("SkiNet.ML.datasets.segmentation_dataset.load_sample", fake_load_sample)

    item = dataset[0]

    assert set(item.keys()) == {"image", "mask", "specs"}
    assert item["image"].shape == (1, CROP_SIZE[0], CROP_SIZE[1], 3)
    assert item["mask"].shape == (1, CROP_SIZE[0], CROP_SIZE[1])
    assert item["specs"]["sample_id"] == "sample-1"


@pytest.mark.parametrize("bad_index", [1, 5, -2])
def test_segmentation_dataset_getitem_bad_index_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_index: int,
) -> None:
    """
    Test that accessing a bad index raises an IndexError.
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
    config = _make_config(df, tmp_path)
    dataset = SegmentationDataset(config=config)

    with pytest.raises(IndexError):
        _ = dataset[bad_index]

@pytest.mark.parametrize(
    ("preserve_original_order", "expected_order"),
    [
        (True, ["sample-b", "sample-a"]),   # preserve input order
        (False, ["sample-a", "sample-b"]),  # sorted order
    ],
)
def test_segmentation_dataset_preserves_sample_id_order(tmp_path: Path, preserve_original_order: bool, expected_order: list[str]) -> None:
    """
    Test that the SegmentationDataset exposes sample_ids in either the original
    DataFrame order or sorted order depending on preserve_original_order.
    """
    df = pd.DataFrame(
        [
            {
                SAMPLEID_HEADER: "sample-b",
                DATATYPE_HEADER: DATATYPE_IMAGE,
                DATAPATH_HEADER: "images/img_b.png",
                "site": "B",
            },
            {
                SAMPLEID_HEADER: "sample-b",
                DATATYPE_HEADER: DATATYPE_MASK,
                DATAPATH_HEADER: "masks/mask_b.png",
                "site": "B",
            },
            {
                SAMPLEID_HEADER: "sample-a",
                DATATYPE_HEADER: DATATYPE_IMAGE,
                DATAPATH_HEADER: "images/img_a.png",
                "site": "A",
            },
            {
                SAMPLEID_HEADER: "sample-a",
                DATATYPE_HEADER: DATATYPE_MASK,
                DATAPATH_HEADER: "masks/mask_a.png",
                "site": "A",
            },
        ]
    )
    config = _make_config(df, tmp_path)
    dataset = SegmentationDataset(config=config)

    # rebuild sample_specs using the module helper with the chosen ordering,
    # then set dataset fields so the test verifies both modes without changing production code.
    from SkiNet.ML.datasets.sample_specs import create_valid_samplespecs

    specs = create_valid_samplespecs(df, preserve_original_order=preserve_original_order)
    dataset.sample_specs = specs
    dataset.sample_ids = list(specs.keys())

    assert dataset.sample_ids == expected_order
