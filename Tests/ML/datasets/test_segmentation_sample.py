from SkiNet.Utils.csv_headers import DATAPATH_HEADER, DATATYPE_HEADER, DATATYPE_IMAGE, DATATYPE_MASK, SAMPLEID_HEADER
from SkiNet.ML.transformations.transform_adapters import AlbumentationsSampleTransform
from SkiNet.ML.datasets.segmentation_dataset import SegmentationDataset
from SkiNet.ML.datasets.sample_specs import Sample
from SkiNet.ML.utils.model_utils import MLWorkflowState
from torchvision.io import write_png
import torch
import pandas as pd
import numpy as np
import albumentations as A
from pathlib import Path
import pytest


CROP_SIZE = (512, 512)  # as per augmentations config
IMG_SIZE = (572, 765)  # as in main_run dummy sample
IMG_CHANNELS = 3  # as in main_run dummy sample
MASK_SIZE = (572, 765)  # as in main_run dummy sample


class IdentityTransform:
    pipeline = A.Compose([])
    visualization_pipeline = None
    expects_tensor_output = True

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
    dataset = SegmentationDataset(
        data_root=tmp_path,
        dataframe=df,
        transform=IdentityTransform(),
        mode=MLWorkflowState.TRAIN,
        cache_in_ram=False,
    )

    assert len(dataset) == expected_len


def test_segmentation_dataset_get_raw_sample_preserves_loaded_shape_and_dtype(tmp_path: Path) -> None:
    """
    get_raw_sample() should return the untransformed sample with the original
    tensor shapes and dtypes produced by the current disk loader (CHW, uint8 for both image and mask).

    For decode_image used to load data items, this is CHW and uint8"""
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
    dataset = SegmentationDataset(
        data_root=tmp_path,
        dataframe=df,
        transform=IdentityTransform(),
        mode=MLWorkflowState.TRAIN,
        cache_in_ram=False,
    )

    raw_sample = dataset.get_raw_sample(0)  # CHW, uint8

    # decode_image returns (C, H, W) where C=3 for RGB images
    assert raw_sample.image.shape == (IMG_CHANNELS, IMG_SIZE[0], IMG_SIZE[1])
    # decode_image returns (C, H, W) where C=1 for grayscale masks
    assert raw_sample.mask.shape == (1, MASK_SIZE[0], MASK_SIZE[1])
    assert raw_sample.image.dtype == torch.uint8
    assert raw_sample.mask.dtype == torch.uint8
    assert raw_sample.specs.sample_id == "sample-1"


@pytest.mark.parametrize(
    "transform,image_shape,mask_shape,image_type,mask_type,dtype_image,dtype_mask",
    [
        (
            AlbumentationsSampleTransform(
                pipeline=A.Compose([A.CenterCrop(height=CROP_SIZE[0], width=CROP_SIZE[1])]),
                expects_tensor_output=False,  # no ToTensorV2, so output should be numpy arrays in HWC/HW format as below
            ),
            (CROP_SIZE[0], CROP_SIZE[1], IMG_CHANNELS),
            (CROP_SIZE[0], CROP_SIZE[1], 1),
            np.ndarray,
            np.ndarray,
            np.uint8,
            np.uint8,
        ),
        (
            AlbumentationsSampleTransform(
                pipeline=A.Compose(
                    [A.CenterCrop(height=CROP_SIZE[0], width=CROP_SIZE[1]),
                     A.ToTensorV2(transpose_mask=True)]
                ),
                expects_tensor_output=True,  # ToTensorV2 with transpose_mask=True should produce torch tensors in CHW format
            ),
            (IMG_CHANNELS, CROP_SIZE[0], CROP_SIZE[1]),
            (1, CROP_SIZE[0], CROP_SIZE[1]),
            torch.Tensor,
            torch.Tensor,
            torch.uint8,  # no Normalize is used, hence images remain uint8
            torch.uint8,
        ),
        (
            AlbumentationsSampleTransform(
                pipeline=A.Compose(
                    [A.CenterCrop(height=CROP_SIZE[0], width=CROP_SIZE[1]),
                     A.Normalize(normalization="image_per_channel", p=1.0), A.ToTensorV2(transpose_mask=True)]
                ),
                expects_tensor_output=True,  # ToTensorV2 with transpose_mask=True should produce torch tensors in CHW format
            ),
            (IMG_CHANNELS, CROP_SIZE[0], CROP_SIZE[1]),
            (1, CROP_SIZE[0], CROP_SIZE[1]),
            torch.Tensor,
            torch.Tensor,
            torch.float32,  # Normalize produces float32 images, but masks remain uint8 even after ToTensorV2 with transpose_mask=True
            torch.uint8,
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
    dtype_image: torch.dtype | np.dtype,
    dtype_mask: torch.dtype | np.dtype
) -> None:
    """
    Test if SegmentationDataset.__getitem__()  preserves the transform output layout:

    - Albumentations without ToTensorV2 returns numpy HWC/HW data
    - Pipelines ending in ToTensorV2 return torch tensors in CHW layout
    - Images should have 3 channels and masks should have 1 channel
    - Data type depends on whether Normalize is used in the transform pipeline or not
    - The shape is set by the CenterCrop in the transform pipeline
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
    dataset = SegmentationDataset(
        data_root=tmp_path,
        dataframe=df,
        transform=transform,
        mode=MLWorkflowState.TRAIN,
        cache_in_ram=False,
    )

    item = dataset[0]

    assert set(item.keys()) == {"image", "mask", "specs"}
    assert isinstance(item["image"], image_type)
    assert isinstance(item["mask"], mask_type)
    assert item["image"].shape == image_shape
    assert item["mask"].shape == mask_shape
    assert item["image"].dtype == dtype_image
    assert item["mask"].dtype == dtype_mask
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
    dataset = SegmentationDataset(
        data_root=tmp_path,
        dataframe=df,
        transform=IdentityTransform(),
        mode=MLWorkflowState.TRAIN,
        cache_in_ram=False,
    )

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
    dataset = SegmentationDataset(
        data_root=tmp_path,
        dataframe=df,
        transform=IdentityTransform(),
        mode=MLWorkflowState.TRAIN,
        cache_in_ram=False,
    )

    from SkiNet.ML.datasets.sample_specs import create_valid_samplespecs

    specs = create_valid_samplespecs(df, preserve_original_order=preserve_original_order)
    dataset.sample_specs = specs
    dataset.sample_ids = list(specs.keys())

    assert dataset.sample_ids == expected_order


def test_segmentation_dataset_cache_in_ram_serves_same_item_as_no_cache(tmp_path: Path) -> None:
    """
    cache_in_ram=True should produce identical __getitem__ output to cache_in_ram=False,
    confirming that the cache path is exercised and doesn't alter the returned data.
    """
    image_rel_path, mask_rel_path = _write_png_sample_files(tmp_path)

    df = pd.DataFrame(
        [
            {SAMPLEID_HEADER: "sample-1", DATATYPE_HEADER: DATATYPE_IMAGE, DATAPATH_HEADER: image_rel_path, "site": "A"},
            {SAMPLEID_HEADER: "sample-1", DATATYPE_HEADER: DATATYPE_MASK, DATAPATH_HEADER: mask_rel_path, "site": "A"},
        ]
    )

    transform = AlbumentationsSampleTransform(
        pipeline=A.Compose([A.CenterCrop(height=CROP_SIZE[0], width=CROP_SIZE[1]), A.ToTensorV2(transpose_mask=True)]),
        expects_tensor_output=True,
    )

    cached = SegmentationDataset(data_root=tmp_path, dataframe=df, transform=transform,
                                 mode=MLWorkflowState.TRAIN, cache_in_ram=True)
    uncached = SegmentationDataset(data_root=tmp_path, dataframe=df, transform=transform,
                                   mode=MLWorkflowState.TRAIN, cache_in_ram=False)

    item_cached = cached[0]
    item_uncached = uncached[0]

    assert torch.equal(item_cached["image"], item_uncached["image"])
    assert torch.equal(item_cached["mask"], item_uncached["mask"])
    assert item_cached["specs"] == item_uncached["specs"]


def test_segmentation_dataset_cache_is_populated_on_init(tmp_path: Path) -> None:
    """
    When cache_in_ram=True, _cache should be a dict keyed by sample_id after __init__,
    containing pre-loaded Sample objects.
    """
    image_rel_path, mask_rel_path = _write_png_sample_files(tmp_path)

    df = pd.DataFrame(
        [
            {SAMPLEID_HEADER: "sample-1", DATATYPE_HEADER: DATATYPE_IMAGE, DATAPATH_HEADER: image_rel_path, "site": "A"},
            {SAMPLEID_HEADER: "sample-1", DATATYPE_HEADER: DATATYPE_MASK, DATAPATH_HEADER: mask_rel_path, "site": "A"},
        ]
    )

    dataset = SegmentationDataset(
        data_root=tmp_path, dataframe=df, transform=IdentityTransform(), mode=MLWorkflowState.TRAIN, cache_in_ram=True
    )

    assert dataset._cache is not None
    assert list(dataset._cache.keys()) == dataset.sample_ids
    cached_sample = dataset._cache["sample-1"]
    assert isinstance(cached_sample.image, torch.Tensor)
    assert cached_sample.image.dtype == torch.uint8


def test_segmentation_dataset_no_cache_is_none(tmp_path: Path) -> None:
    """When cache_in_ram=False, _cache should remain None."""
    df = pd.DataFrame(
        [
            {SAMPLEID_HEADER: "sample-1", DATATYPE_HEADER: DATATYPE_IMAGE, DATAPATH_HEADER: "images/img1.png", "site": "A"},
            {SAMPLEID_HEADER: "sample-1", DATATYPE_HEADER: DATATYPE_MASK, DATAPATH_HEADER: "masks/mask1.png", "site": "A"},
        ]
    )

    dataset = SegmentationDataset(
        data_root=tmp_path, dataframe=df, transform=IdentityTransform(), mode=MLWorkflowState.TRAIN, cache_in_ram=False
    )

    assert dataset._cache is None
