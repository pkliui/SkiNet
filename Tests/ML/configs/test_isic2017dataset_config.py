import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import SkiNet.ML.configs.data_configs.base_data_config as base_data_config
from SkiNet.ML.configs.data_configs.isic2017dataset_config.isic2017dataset_config import (
    ISIC2017DatasetConfig,
    ISIC2017StratificationOptions,
)
from SkiNet.Utils.csv_headers import (
    DATAPATH_HEADER,
    DATATYPE_HEADER,
    ISIC2017_MELANOMA_HEADER,
    ISIC2017_PREDEFINED_SPLIT_HEADER,
    ISIC2017_SEBORRHEIC_KERATOSIS_HEADER,
    SAMPLEID_HEADER,
)
from SkiNet.Utils.project_paths import ISIC2017_CSV_NAME


def _write_csv(tmp_path: Path, columns: list[str], value: str = "dummy") -> None:
    df = pd.DataFrame([{col: value for col in columns}])
    (tmp_path / ISIC2017_CSV_NAME).write_text(df.to_csv(index=False))


# -----------------------------------------------------------------------
# Column validation
# -----------------------------------------------------------------------

@pytest.mark.parametrize(
    "columns,should_raise",
    [
        ([SAMPLEID_HEADER, DATAPATH_HEADER, DATATYPE_HEADER], False),
        ([SAMPLEID_HEADER, DATAPATH_HEADER], True),
        ([DATAPATH_HEADER, DATATYPE_HEADER], True),
        ([SAMPLEID_HEADER, DATATYPE_HEADER], True),
        ([SAMPLEID_HEADER], True),
    ],
)
def test_isic2017datasetconfig_column_validation(
    tmp_path: Path, columns: list[str], should_raise: bool
) -> None:
    _write_csv(tmp_path, columns)
    cfg = ISIC2017DatasetConfig(local_data_root=str(tmp_path))  # type: ignore
    if should_raise:
        with pytest.raises(ValueError, match="is missing required columns"):
            _ = cfg.metadata
    else:
        _ = cfg.metadata


def test_isic2017datasetconfig_empty_values(tmp_path: Path) -> None:
    _write_csv(tmp_path, [SAMPLEID_HEADER, DATAPATH_HEADER, DATATYPE_HEADER], value="")
    cfg = ISIC2017DatasetConfig(local_data_root=str(tmp_path))  # type: ignore
    with pytest.raises(ValueError, match="all required columns have only empty values"):
        _ = cfg.metadata


def test_isic2017datasetconfig_empty_csv(tmp_path: Path) -> None:
    (tmp_path / ISIC2017_CSV_NAME).write_text("")
    cfg = ISIC2017DatasetConfig(local_data_root=str(tmp_path))  # type: ignore
    with pytest.raises(pd.errors.EmptyDataError):
        _ = cfg.metadata


# -----------------------------------------------------------------------
# File-path loading (local)
# -----------------------------------------------------------------------

def test_isic2017datasetconfig_file_paths_exist(tmp_path: Path) -> None:
    img_path = tmp_path / "img.jpg"
    mask_path = tmp_path / "mask.png"
    img_path.write_text("image")
    mask_path.write_text("mask")

    rows = [
        {SAMPLEID_HEADER: "ISIC_0000001", DATAPATH_HEADER: str(img_path), DATATYPE_HEADER: "image"},
        {SAMPLEID_HEADER: "ISIC_0000001", DATAPATH_HEADER: str(mask_path), DATATYPE_HEADER: "mask"},
    ]
    pd.DataFrame(rows).to_csv(tmp_path / ISIC2017_CSV_NAME, index=False)

    cfg = ISIC2017DatasetConfig(local_data_root=str(tmp_path))  # type: ignore
    df = cfg.metadata

    for i, row in enumerate(rows):
        loaded = df.iloc[i]
        assert str(loaded[SAMPLEID_HEADER]) == row[SAMPLEID_HEADER]
        assert str(loaded[DATAPATH_HEADER]) == row[DATAPATH_HEADER]
        assert str(loaded[DATATYPE_HEADER]) == row[DATATYPE_HEADER]
        assert os.path.exists(loaded[DATAPATH_HEADER])


# -----------------------------------------------------------------------
# get_split_config
# -----------------------------------------------------------------------

def test_get_split_config_default_stratify_column() -> None:
    cfg = ISIC2017DatasetConfig(local_data_root="/dummy")  # type: ignore
    sc = cfg.get_split_config()
    assert sc.stratify_column == ISIC2017_MELANOMA_HEADER


def test_get_split_config_seborrheic_keratosis_stratify_column() -> None:
    cfg = ISIC2017DatasetConfig(
        local_data_root="/dummy",
        split_stratify_column=ISIC2017StratificationOptions.ISIC2017_SEBORRHEIC_KERATOSIS,
    )  # type: ignore
    sc = cfg.get_split_config()
    assert sc.stratify_column == ISIC2017_SEBORRHEIC_KERATOSIS_HEADER


def test_get_split_config_no_stratification() -> None:
    cfg = ISIC2017DatasetConfig(local_data_root="/dummy", split_stratify_column=None)  # type: ignore
    sc = cfg.get_split_config()
    assert sc.stratify_column is None


def test_get_split_config_sizes_forwarded() -> None:
    cfg = ISIC2017DatasetConfig(
        local_data_root="/dummy",
        split_train_size=0.7,
        split_val_size=0.2,
        split_test_size=0.1,
        split_random_seed=99,
    )  # type: ignore
    sc = cfg.get_split_config()
    assert sc.train_size == 0.7
    assert sc.val_size == 0.2
    assert sc.test_size == 0.1
    assert sc.random_seed == 99


# -----------------------------------------------------------------------
# predefined_split_column
# -----------------------------------------------------------------------

def test_predefined_split_column_defaults_to_isic2017_header() -> None:
    """ISIC2017 uses official challenge splits by default."""
    cfg = ISIC2017DatasetConfig(local_data_root="/dummy")  # type: ignore
    assert cfg.predefined_split_column == ISIC2017_PREDEFINED_SPLIT_HEADER


def test_predefined_split_column_can_be_disabled() -> None:
    """Setting predefined_split_column=None falls back to random splitting."""
    cfg = ISIC2017DatasetConfig(local_data_root="/dummy", predefined_split_column=None)  # type: ignore
    assert cfg.predefined_split_column is None


# -----------------------------------------------------------------------
# StratificationOptions enum
# -----------------------------------------------------------------------

def test_isic2017stratification_options_values() -> None:
    assert ISIC2017StratificationOptions.ISIC2017_MELANOMA.value == ISIC2017_MELANOMA_HEADER
    assert ISIC2017StratificationOptions.ISIC2017_SEBORRHEIC_KERATOSIS.value == ISIC2017_SEBORRHEIC_KERATOSIS_HEADER


# -----------------------------------------------------------------------
# Azure (mocked)
# -----------------------------------------------------------------------

def test_isic2017datasetconfig_file_paths_exist_azure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    img_path = tmp_path / "img.jpg"
    mask_path = tmp_path / "mask.png"
    img_path.write_text("image")
    mask_path.write_text("mask")

    rows = [
        {SAMPLEID_HEADER: "ISIC_0000001", DATAPATH_HEADER: str(img_path), DATATYPE_HEADER: "image"},
        {SAMPLEID_HEADER: "ISIC_0000001", DATAPATH_HEADER: str(mask_path), DATATYPE_HEADER: "mask"},
    ]
    # Write the CSV to the mount point so data_root (= tmp_path / "") resolves correctly
    pd.DataFrame(rows).to_csv(tmp_path / ISIC2017_CSV_NAME, index=False)

    class MockAzureSetup:
        @staticmethod
        def service_principal_authentication() -> None:
            return None

        @staticmethod
        def get_azureml_filesystem(dataset_name: str) -> Any:
            return None

        @staticmethod
        def get_azure_uri(dataset_name: str) -> tuple[str, str]:
            return ("unused", "mock/path/on/azure")

        @staticmethod
        def from_yaml(path: str) -> Any:
            class DummyConfig:
                PATH_ON_DATASTORE = {"ISIC2017_DATASET": ""}
            return DummyConfig()

    monkeypatch.setattr(base_data_config, "AzureSetup", MockAzureSetup)

    cfg = ISIC2017DatasetConfig(azure_data=True, azure_blob_mount_point=str(tmp_path), kind="isic2017")  # type: ignore
    df = cfg.metadata

    for i, row in enumerate(rows):
        loaded = df.iloc[i]
        assert str(loaded[SAMPLEID_HEADER]) == row[SAMPLEID_HEADER]
        assert str(loaded[DATAPATH_HEADER]) == row[DATAPATH_HEADER]
        assert os.path.exists(loaded[DATAPATH_HEADER])
