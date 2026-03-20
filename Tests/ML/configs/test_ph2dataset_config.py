# test_ph2dataset_config.py
import os
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import SkiNet.ML.configs.data_configs.base_data_config as base_data_config
from SkiNet.ML.configs.data_configs.ph2dataset_config.ph2dataset_config import PH2DatasetConfig
from SkiNet.Utils.csv_headers import DATAPATH_HEADER, DATATYPE_HEADER, SAMPLEID_HEADER
from SkiNet.Utils.project_paths import PH2_CSV_NAME


@pytest.mark.parametrize(
    "columns,should_raise",
    [
        ([SAMPLEID_HEADER, DATAPATH_HEADER, DATATYPE_HEADER], False),  # all present
        ([SAMPLEID_HEADER, DATAPATH_HEADER], True),  # missing one
        ([DATAPATH_HEADER, DATATYPE_HEADER], True),  # missing one
        ([SAMPLEID_HEADER, DATATYPE_HEADER], True),  # missing one
        ([SAMPLEID_HEADER], True),  # only one
    ]
)
def test_ph2datasetconfig_column_validation(tmp_path: Path, columns: list[str], should_raise: bool) -> None:
    """
    Test the PH2DatasetConfig column validation when all columns are present and when there are some missing columns.
    """
    df = pd.DataFrame([{col: "dummy" for col in columns}])
    csv_path = tmp_path / PH2_CSV_NAME
    df.to_csv(csv_path, index=False)

    cfg = PH2DatasetConfig(local_data_root=str(tmp_path))   # type: ignore
    if should_raise:
        with pytest.raises(ValueError, match="is missing required columns"):
            _ = cfg.metadata
    else:
        _ = cfg.metadata

@pytest.mark.parametrize(
    "columns,should_raise",
    [
        ([SAMPLEID_HEADER, DATAPATH_HEADER, DATATYPE_HEADER], True),  # all present
    ]
)
def test_ph2datasetconfig_empty_values(tmp_path: Path, columns: list[str], should_raise: bool) -> None:
    """
    Test the PH2DatasetConfig column validation when all columns are present but their values are empty strings.
    """
    df = pd.DataFrame([{col: "" for col in columns}])
    csv_path = tmp_path / PH2_CSV_NAME
    df.to_csv(csv_path, index=False)

    cfg = PH2DatasetConfig(local_data_root=str(tmp_path))  # type: ignore
    if should_raise:
        with pytest.raises(ValueError, match="all required columns have only empty values"):
            _ = cfg.metadata
    else:
        _ = cfg.metadata


def test_ph2datasetconfig_empty_csv(tmp_path: Path) -> None:
    """
    Test the PH2DatasetConfig column validation when the CSV is empty, i.e. no rows and no columns.
    """
    csv_path = tmp_path / PH2_CSV_NAME
    csv_path.write_text("")

    cfg = PH2DatasetConfig(local_data_root=str(tmp_path))  # type: ignore
    with pytest.raises(pd.errors.EmptyDataError):
        _ = cfg.metadata


def test_ph2datasetconfig_file_paths_exist(tmp_path: Path) -> None:
    """
    Test that all file paths referenced in the loaded DataFrame exist.
    """
    # Create dummy files
    img_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.png"
    img_path.write_text("image data")
    mask_path.write_text("mask data")

    # Expected data
    expected_rows = [
        {SAMPLEID_HEADER: "ID1", DATAPATH_HEADER: str(img_path), DATATYPE_HEADER: "image"},
        {SAMPLEID_HEADER: "ID1", DATAPATH_HEADER: str(mask_path), DATATYPE_HEADER: "mask"},
    ]

    # Create CSV referencing those files
    df = pd.DataFrame(expected_rows)
    csv_path = tmp_path / PH2_CSV_NAME
    df.to_csv(csv_path, index=False)

    cfg = PH2DatasetConfig(local_data_root=str(tmp_path))  # type: ignore
    df_loaded = cfg.metadata

    # Check that all paths exist and all columns match expected values
    for i, row in enumerate(expected_rows):
        loaded_row = df_loaded.iloc[i]
        assert str(loaded_row[SAMPLEID_HEADER]) == str(row[SAMPLEID_HEADER])
        assert str(loaded_row[DATAPATH_HEADER]) == str(row[DATAPATH_HEADER])
        assert str(loaded_row[DATATYPE_HEADER]) == str(row[DATATYPE_HEADER])
        assert os.path.exists(loaded_row[DATAPATH_HEADER])


# ---------------------------------------------------Azure-------------------------------------------------#

def test_ph2datasetconfig_file_paths_exist_azure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Test that all file paths referenced in the loaded DataFrame exist (Azure scenario, mocked).
    """
    # Create dummy files
    img_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.png"
    img_path.write_text("image data")
    mask_path.write_text("mask data")

#    Create the expected metadata CSV file
    metadata_csv_path = tmp_path / "ph2_metadata.csv"
    expected_rows = [
        {SAMPLEID_HEADER: "ID1", DATAPATH_HEADER: str(img_path), DATATYPE_HEADER: "image"},
        {SAMPLEID_HEADER: "ID1", DATAPATH_HEADER: str(mask_path), DATATYPE_HEADER: "mask"},
    ]

    df = pd.DataFrame(expected_rows)
    df.to_csv(metadata_csv_path, index=False)

    # Mock AzureSetup and fs.open to return our CSV buffer
    class MockFS:
        def open(self, *args: Any, **kwargs: Any) -> StringIO:
            return StringIO(metadata_csv_path.read_text())

    class MockAzureSetup:
        @staticmethod
        def service_principal_authentication() -> None:
            return None

        @staticmethod
        def get_azureml_filesystem(dataset_name: str) -> MockFS:
            return MockFS()

        @staticmethod
        def get_azure_uri(dataset_name: str) -> tuple[str, str]:
            return ("unused", "mock/path/on/azure")

        @staticmethod
        def from_yaml(path: str) -> Any:
            # Return a dummy config object with PATH_ON_DATASTORE attribute
            class DummyConfig:
                PATH_ON_DATASTORE = {"PH2_DATASET": ""}
            return DummyConfig()

    # Mock the AzureSetup class in base_data_config
    monkeypatch.setattr(base_data_config, "AzureSetup", MockAzureSetup)

    cfg = PH2DatasetConfig(azure_data=True,
                           azure_blob_mount_point=str(tmp_path),
                           kind="ph2")  # type: ignore
    df_loaded = cfg.metadata

    # Check that all paths exist and all columns match expected values
    for i, row in enumerate(expected_rows):
        loaded_row = df_loaded.iloc[i]
        assert str(loaded_row[SAMPLEID_HEADER]) == str(row[SAMPLEID_HEADER])
        assert str(loaded_row[DATAPATH_HEADER]) == str(row[DATAPATH_HEADER])
        assert str(loaded_row[DATATYPE_HEADER]) == str(row[DATATYPE_HEADER])
        assert os.path.exists(loaded_row[DATAPATH_HEADER])
