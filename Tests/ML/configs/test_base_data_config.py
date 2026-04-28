from copy import deepcopy
from pathlib import Path
from typing import ClassVar, Optional

import pandas as pd
import pytest

from SkiNet.ML.configs.data_configs.base_data_config import BaseDataConfig
from SkiNet.Utils.experiment_keys import DatasetKey

CSV_NAME = "dummy.csv"

class DummyConfig(BaseDataConfig):
    REQUIRED_COLUMNS: ClassVar[frozenset[str]] = frozenset({"a", "b"})
    DATASET_KEY: ClassVar[Optional[DatasetKey]] = None
    METADATA_CSV_NAME: ClassVar[str] = CSV_NAME

@pytest.mark.parametrize(
    "columns,should_raise",
    [
        (["a", "b"], False),  # all present
        (["a"], True),        # missing one
        (["b"], True),        # missing one
    ]
)
def test_basedataconfig_column_validation(tmp_path: Path, columns: list[str], should_raise: bool) -> None:
    """
    Test the column validation logic in BaseDataConfig.
    """
    df = pd.DataFrame([{col: "dummy" for col in columns}])
    csv_path = tmp_path / CSV_NAME
    df.to_csv(csv_path, index=False)
    cfg = DummyConfig(local_data_root=str(tmp_path), azure_data=False, azure_blob_mount_point=str(tmp_path))
    if should_raise:
        with pytest.raises(ValueError, match="missing required columns"):
            _ = cfg.metadata
    else:
        _ = cfg.metadata


def test_basedataconfig_all_empty_required_columns(tmp_path: Path) -> None:
    """
    Test the behavior when all required columns are empty.
    """
    df = pd.DataFrame([{"a": "", "b": ""}])
    csv_path = tmp_path / CSV_NAME
    df.to_csv(csv_path, index=False)
    cfg = DummyConfig(local_data_root=str(tmp_path), azure_data=False, azure_blob_mount_point=str(tmp_path))
    with pytest.raises(ValueError, match="all required columns have only empty values"):
        _ = cfg.metadata


def test_basedataconfig_empty_csv(tmp_path: Path) -> None:
    """
    Test the behavior when the CSV file is empty.
    """
    csv_path = tmp_path / CSV_NAME
    csv_path.write_text("")
    cfg = DummyConfig(local_data_root=str(tmp_path), azure_data=False, azure_blob_mount_point=str(tmp_path))
    with pytest.raises(pd.errors.EmptyDataError):
        _ = cfg.metadata

def test_basedataconfig_success(tmp_path: Path) -> None:
    """
    Test the successful loading of a valid CSV file.
    """
    df = pd.DataFrame([{"a": 1, "b": 2}])
    csv_path = tmp_path / CSV_NAME
    df.to_csv(csv_path, index=False)
    cfg = DummyConfig(local_data_root=str(tmp_path), azure_data=False, azure_blob_mount_point=str(tmp_path))
    loaded = cfg.metadata
    assert set(loaded.columns) == {"a", "b"}
    assert loaded.iloc[0]["a"] == 1
    assert loaded.iloc[0]["b"] == 2


def test_basedataconfig_metadata_returns_copy(tmp_path: Path) -> None:
    """
    Mutating the DataFrame returned from metadata must not mutate the cached metadata.
    """
    df = pd.DataFrame([{"a": 1, "b": 2}])
    csv_path = tmp_path / CSV_NAME
    df.to_csv(csv_path, index=False)
    cfg = DummyConfig(local_data_root=str(tmp_path), azure_data=False, azure_blob_mount_point=str(tmp_path))

    loaded = cfg.metadata
    loaded.loc[0, "a"] = 999

    assert cfg.metadata.iloc[0]["a"] == 1


def test_basedataconfig_deepcopy_clears_metadata_cache(tmp_path: Path) -> None:
    """
    deepcopy must reset _metadata to None so each copy lazy-loads independently,
    without duplicating the same DataFrame across multiple trial configs.
    """
    df = pd.DataFrame([{"a": 1, "b": 2}])
    csv_path = tmp_path / CSV_NAME
    df.to_csv(csv_path, index=False)

    original = DummyConfig(local_data_root=str(tmp_path), azure_data=False, azure_blob_mount_point=str(tmp_path))
    _ = original.metadata  # populate the cache

    copy = deepcopy(original)

    assert copy._metadata is None, "deepcopy must not carry the cached DataFrame"
    assert original._metadata is not None, "original cache must be unaffected"

    loaded = copy.metadata  # copy must still lazy-load correctly
    assert set(loaded.columns) == {"a", "b"}
    assert loaded.iloc[0]["a"] == 1
