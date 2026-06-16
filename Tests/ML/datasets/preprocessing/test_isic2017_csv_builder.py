import argparse
from pathlib import Path

import pandas as pd
import pytest

from SkiNet.ML.datasets.preprocessing.isic2017_csv_builder import (
    ISIC2017BaseCSVBuilder,
    ISIC2017LocalCSVBuilder,
)
from SkiNet.Utils.csv_headers import (
    DATAPATH_HEADER,
    DATATYPE_HEADER,
    ISIC2017_MELANOMA_HEADER,
    ISIC2017_PREDEFINED_SPLIT_HEADER,
    ISIC2017_SEBORRHEIC_KERATOSIS_HEADER,
    SAMPLEID_HEADER,
)
from SkiNet.Utils.project_paths import (
    ISIC2017_CSV_NAME,
    ISIC2017_TEST_GT_CSV_NAME,
    ISIC2017_TRAIN_GT_CSV_NAME,
    ISIC2017_VAL_GT_CSV_NAME,
)


# -----------------------------------------------------------------------
# Concrete stub for abstract ISIC2017BaseCSVBuilder
# -----------------------------------------------------------------------

class _StubISIC2017Builder(ISIC2017BaseCSVBuilder):
    """Minimal concrete subclass for testing base methods."""

    def __init__(self, data_root: str = "/dummy") -> None:
        super().__init__()
        self._data_root = data_root

    @property
    def data_root(self) -> str:
        return self._data_root

    @property
    def image_pattern(self) -> str:
        return "**/*.jpg"

    @property
    def mask_pattern(self) -> str:
        return "**/*_segmentation.png"

    def datapath_func(self, path: str) -> str:
        return path

    def load_diagnosis_csv(self, csv_name: str) -> pd.DataFrame:
        return pd.DataFrame()

    def create_metadata_csv(self) -> None:
        pass


# -----------------------------------------------------------------------
# sampleid_func
# -----------------------------------------------------------------------

@pytest.mark.parametrize(
    "path,expected",
    [
        ("/root/ISIC-2017_Training_Data/ISIC_0000000.jpg", "ISIC_0000000"),
        ("/root/ISIC-2017_Training_Part1_GroundTruth/ISIC_0000000_segmentation.png", "ISIC_0000000"),
        ("ISIC-2017_Test_v2_Data/ISIC_0012345.jpg", "ISIC_0012345"),
        ("ISIC-2017_Validation_Part1_GroundTruth/ISIC_0099999_segmentation.png", "ISIC_0099999"),
    ],
)
def test_sampleid_func(path: str, expected: str) -> None:
    builder = _StubISIC2017Builder()
    assert builder.sampleid_func(path) == expected


# -----------------------------------------------------------------------
# predefined_split_func
# -----------------------------------------------------------------------

@pytest.mark.parametrize(
    "path,expected_split",
    [
        ("/data/ISIC-2017_Training_Data/ISIC_0000000.jpg", "train"),
        ("/data/ISIC-2017_Validation_Data/ISIC_0000001.jpg", "val"),
        ("/data/ISIC-2017_Test_v2_Data/ISIC_0000002.jpg", "test"),
        ("/data/ISIC-2017_Training_Part1_GroundTruth/ISIC_0000000_segmentation.png", "train"),
        ("/data/ISIC-2017_Validation_Part1_GroundTruth/ISIC_0000001_segmentation.png", "val"),
        ("/data/ISIC-2017_Test_v2_Part1_GroundTruth/ISIC_0000002_segmentation.png", "test"),
        ("/data/unknown_dir/ISIC_0000003.jpg", "unknown"),
    ],
)
def test_predefined_split_func(path: str, expected_split: str) -> None:
    builder = _StubISIC2017Builder()
    assert builder.predefined_split_func(path) == expected_split


# -----------------------------------------------------------------------
# load_all_diagnosis_data — loads and concatenates three CSVs
# -----------------------------------------------------------------------

def test_load_all_diagnosis_data_concatenates(tmp_path: Path) -> None:
    for csv_name, label in [
        (ISIC2017_TRAIN_GT_CSV_NAME, "train"),
        (ISIC2017_VAL_GT_CSV_NAME, "val"),
        (ISIC2017_TEST_GT_CSV_NAME, "test"),
    ]:
        pd.DataFrame([{"image_id": f"ISIC_{label}", ISIC2017_MELANOMA_HEADER: 0,
                       ISIC2017_SEBORRHEIC_KERATOSIS_HEADER: 1}]).to_csv(tmp_path / csv_name, index=False)

    args = argparse.Namespace(local_data_root=str(tmp_path))
    builder = ISIC2017LocalCSVBuilder(args)
    df = builder.load_all_diagnosis_data()

    assert len(df) == 3
    assert set(df["image_id"].tolist()) == {"ISIC_train", "ISIC_val", "ISIC_test"}


# -----------------------------------------------------------------------
# create_merged_isic2017_metadata — integration of base builder
# -----------------------------------------------------------------------

def _make_gt_csvs(tmp_path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    for csv_name in (ISIC2017_TRAIN_GT_CSV_NAME, ISIC2017_VAL_GT_CSV_NAME, ISIC2017_TEST_GT_CSV_NAME):
        df.to_csv(tmp_path / csv_name, index=False)


def _make_isic2017_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """
    Create minimal ISIC2017 directory structure matching the local glob patterns:
      ISIC-2017_*_Data/*/*.jpg
      ISIC-2017_*_Part1_GroundTruth/*/*_segmentation.png
    """
    # image: ISIC-2017_Training_Data/<subdir>/ISIC_0000001.jpg
    train_data = tmp_path / "ISIC-2017_Training_Data" / "subdir"
    train_data.mkdir(parents=True)
    img = train_data / "ISIC_0000001.jpg"
    img.write_text("image")

    # mask: ISIC-2017_Training_Part1_GroundTruth/<subdir>/ISIC_0000001_segmentation.png
    train_gt = tmp_path / "ISIC-2017_Training_Part1_GroundTruth" / "subdir"
    train_gt.mkdir(parents=True)
    mask = train_gt / "ISIC_0000001_segmentation.png"
    mask.write_text("mask")

    return img, mask


def test_create_merged_isic2017_metadata_adds_predefined_split_and_diagnosis(tmp_path: Path) -> None:
    _make_isic2017_dirs(tmp_path)
    gt_rows = [{"image_id": "ISIC_0000001", ISIC2017_MELANOMA_HEADER: 1, ISIC2017_SEBORRHEIC_KERATOSIS_HEADER: 0}]
    _make_gt_csvs(tmp_path, gt_rows)

    args = argparse.Namespace(local_data_root=str(tmp_path))
    builder = ISIC2017LocalCSVBuilder(args)
    df = builder.create_merged_isic2017_metadata()

    assert ISIC2017_PREDEFINED_SPLIT_HEADER in df.columns
    assert ISIC2017_MELANOMA_HEADER in df.columns
    assert ISIC2017_SEBORRHEIC_KERATOSIS_HEADER in df.columns

    image_rows = df[df[DATATYPE_HEADER] == "image"]
    assert len(image_rows) >= 1
    assert (image_rows[SAMPLEID_HEADER] == "ISIC_0000001").any()
    assert ISIC2017_PREDEFINED_SPLIT_HEADER in df.columns  # split assignment tested in unit tests above
    assert (image_rows[ISIC2017_MELANOMA_HEADER] == 1).any()


def test_create_merged_isic2017_metadata_csv_written(tmp_path: Path) -> None:
    _make_isic2017_dirs(tmp_path)
    gt_rows = [{"image_id": "ISIC_0000001", ISIC2017_MELANOMA_HEADER: 0, ISIC2017_SEBORRHEIC_KERATOSIS_HEADER: 0}]
    _make_gt_csvs(tmp_path, gt_rows)

    args = argparse.Namespace(local_data_root=str(tmp_path))
    builder = ISIC2017LocalCSVBuilder(args)
    builder.create_metadata_csv()

    csv_path = tmp_path / ISIC2017_CSV_NAME
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert SAMPLEID_HEADER in df.columns
    assert DATAPATH_HEADER in df.columns


# -----------------------------------------------------------------------
# output_csv_name property
# -----------------------------------------------------------------------

def test_output_csv_name() -> None:
    builder = _StubISIC2017Builder()
    assert builder.output_csv_name == ISIC2017_CSV_NAME
