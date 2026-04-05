from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import albumentations as A
import pandas as pd
import pytest
import torch
import numpy as np
from torchvision.io import write_png

from SkiNet.ML.datasets.segmentation_dataset import SegmentationDataset
from SkiNet.ML.datasets.sample_specs import Sample
from SkiNet.ML.transformations.transform_adapters import AlbumentationsSampleTransform
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


class IdentityTransform:
    """
    Test transform that returns the input sample unchanged.
    """

    def __call__(self, sample: Sample) -> Sample:
        return sample


class ResizeToFloatTransform:
    """
    Test transform that returns tensors with controlled output shape and dtype.
    """

    def __init__(
        self,
        image_shape: tuple[int, ...] = (3, CROP_SIZE[0], CROP_SIZE[1]),
        mask_shape: tuple[int, ...] = (1, CROP_SIZE[0], CROP_SIZE[1]),
        image_dtype: torch.dtype = torch.float32,
        mask_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.image_shape = image_shape
        self.mask_shape = mask_shape
        self.image_dtype = image_dtype
        self.mask_dtype = mask_dtype

    def __call__(self, sample: Sample) -> Sample:
        return sample.model_copy(
            update={
                "image": torch.ones(self.image_shape, dtype=self.image_dtype),
                "mask": torch.ones(self.mask_shape, dtype=self.mask_dtype),
            }
        )


def row_factory(sample_id: str, dtype: str, path: str, site: str) -> dict:
    """Row factory: keeps test cases concise."""
    return {
        SAMPLEID_HEADER: sample_id,
        DATATYPE_HEADER: dtype,
        DATAPATH_HEADER: path,
        "site": site,
    }


def _write_png_sample_files(
    data_root: Path,
    image_rel_path: str = "images/img1.png",
    mask_rel_path: str = "masks/mask1.png",
) -> tuple[str, str]:
    """
    Write one RGB image PNG and one single-channel mask PNG under the dataset root for integration-style dataset tests.
    """
    image_path = data_root / image_rel_path
    mask_path = data_root / mask_rel_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    image_array = np.full((IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS), 255, dtype=np.uint8)
    mask_array = np.full((MASK_SIZE[0], MASK_SIZE[1]), 1, dtype=np.uint8)

    write_png(
        torch.from_numpy(image_array).permute(2, 0, 1).to(torch.uint8),
        str(image_path),
    )
    write_png(
        torch.from_numpy(mask_array).unsqueeze(0).to(torch.uint8),
        str(mask_path),
    )

    return image_rel_path, mask_rel_path

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

    dataset = SegmentationDataset(config=config, transform=IdentityTransform())

    assert len(dataset) == expected_len


def test_segmentation_dataset_get_raw_sample_preserves_loaded_shape_and_dtype(tmp_path: Path) -> None:
    """
    get_raw_sample() should return the untransformed sample with the original
    tensor shapes and dtypes produced by the current disk loader.
    """
    image_rel_path, mask_rel_path = _write_png_sample_files(tmp_path)

    df = pd.DataFrame(
        [
            {
                SAMPLEID_HEADER: "sample-1",
                DATATYPE_HEADER: DATATYPE_IMAGE,
                DATAPATH_HEADER: image_rel_path,
                "site": "A",
            },
            {
                SAMPLEID_HEADER: "sample-1",
                DATATYPE_HEADER: DATATYPE_MASK,
                DATAPATH_HEADER: mask_rel_path,
                "site": "A",
            },
        ]
    )
    config = _make_config(df, tmp_path)
    config = cast(Any, config)
    dataset = SegmentationDataset(config=config, transform=IdentityTransform())

    raw_sample = dataset.get_raw_sample(0)

    assert raw_sample.image.shape == (IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS)
    assert raw_sample.mask.shape == (MASK_SIZE[0], MASK_SIZE[1])
    assert raw_sample.image.dtype == torch.uint8
    assert raw_sample.mask.dtype == torch.uint8
    assert raw_sample.specs.sample_id == "sample-1"


@pytest.mark.parametrize(
    "transform,image_shape,mask_shape,image_type,mask_type",
    [
        (
            AlbumentationsSampleTransform(
                pipeline=A.Compose([A.CenterCrop(height=CROP_SIZE[0], width=CROP_SIZE[1])]),
                expects_tensor_output=False,
            ),
            (CROP_SIZE[0], CROP_SIZE[1], IMG_CHANNELS),
            (CROP_SIZE[0], CROP_SIZE[1], 1),
            np.ndarray,
            np.ndarray,
        ),
        (
            AlbumentationsSampleTransform(
                pipeline=A.Compose(
                    [A.CenterCrop(height=CROP_SIZE[0], width=CROP_SIZE[1]), A.ToTensorV2(transpose_mask=True)]
                ),
                expects_tensor_output=True,
            ),
            (IMG_CHANNELS, CROP_SIZE[0], CROP_SIZE[1]),
            (1, CROP_SIZE[0], CROP_SIZE[1]),
            torch.Tensor,
            torch.Tensor,
        ),
    ],
)
def test_segmentation_dataset_getitem_returns_expected_shapes_and_dtypes_after_transform(
    tmp_path: Path,
    transform: AlbumentationsSampleTransform,
    image_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
    image_type: type[np.ndarray] | type[torch.Tensor],
    mask_type: type[np.ndarray] | type[torch.Tensor],
) -> None:
    """
    __getitem__() should preserve the transform output layout:
    Albumentations without ToTensorV2 returns numpy HWC/HW data, while pipelines
    ending in ToTensorV2 return torch tensors in CHW layout.
    """
    image_rel_path, mask_rel_path = _write_png_sample_files(tmp_path)

    df = pd.DataFrame(
        [
            {
                SAMPLEID_HEADER: "sample-1",
                DATATYPE_HEADER: DATATYPE_IMAGE,
                DATAPATH_HEADER: image_rel_path,
                "site": "A",
            },
            {
                SAMPLEID_HEADER: "sample-1",
                DATATYPE_HEADER: DATATYPE_MASK,
                DATAPATH_HEADER: mask_rel_path,
                "site": "A",
            },
        ]
    )
    config = _make_config(df, tmp_path)
    config = cast(Any, config)
    dataset = SegmentationDataset(config=config, transform=transform)

    item = dataset[0]

    assert set(item.keys()) == {"image", "mask", "specs"}
    assert isinstance(item["image"], image_type)
    assert isinstance(item["mask"], mask_type)
    assert item["image"].shape == image_shape
    assert item["mask"].shape == mask_shape
    assert item["specs"]["sample_id"] == "sample-1"


@pytest.mark.parametrize("bad_index", [1, 5, -2])
def test_segmentation_dataset_getitem_bad_index_raises(
    tmp_path: Path,
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
    dataset = SegmentationDataset(config=config, transform=IdentityTransform())

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
    dataset = SegmentationDataset(config=config, transform=IdentityTransform())

    # rebuild sample_specs using the module helper with the chosen ordering,
    # then set dataset fields so the test verifies both modes without changing production code.
    from SkiNet.ML.datasets.sample_specs import create_valid_samplespecs

    specs = create_valid_samplespecs(df, preserve_original_order=preserve_original_order)
    dataset.sample_specs = specs
    dataset.sample_ids = list(specs.keys())

    assert dataset.sample_ids == expected_order
