from pathlib import Path
from typing import ClassVar, Optional, Set

import pandas as pd
import pytest

from SkiNet.ML.configs.base_data_config import BaseDataConfig
from SkiNet.ML.configs.datasets.dataset_keys import AzureDatasetKey


class DummyConfig(BaseDataConfig):
    REQUIRED_COLUMNS: ClassVar[Set[str]] = {"a", "b"}
    AZURE_DATASET_KEY: ClassVar[Optional[AzureDatasetKey]] = None
    AZURE_CSV_NAME: ClassVar[Optional[str]] = None


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
    csv_path = tmp_path / "dummy.csv"
    df.to_csv(csv_path, index=False)
    cfg = DummyConfig(csv_path=str(csv_path))
    if should_raise:
        with pytest.raises(ValueError, match="missing required columns"):
            _ = cfg.data_frame
    else:
        _ = cfg.data_frame


def test_basedataconfig_all_empty_required_columns(tmp_path: Path) -> None:
    """
    Test the behavior when all required columns are empty.
    """
    df = pd.DataFrame([{"a": "", "b": ""}])
    csv_path = tmp_path / "dummy.csv"
    df.to_csv(csv_path, index=False)
    cfg = DummyConfig(csv_path=str(csv_path))
    with pytest.raises(ValueError, match="all required columns have only empty values"):
        _ = cfg.data_frame


def test_basedataconfig_empty_csv(tmp_path: Path) -> None:
    """
    Test the behavior when the CSV file is empty.
    """
    csv_path = tmp_path / "dummy.csv"
    csv_path.write_text("")
    cfg = DummyConfig(csv_path=str(csv_path))
    with pytest.raises(pd.errors.EmptyDataError):
        _ = cfg.data_frame

def test_basedataconfig_success(tmp_path: Path) -> None:
    """
    Test the successful loading of a valid CSV file.
    """
    df = pd.DataFrame([{"a": 1, "b": 2}])
    csv_path = tmp_path / "dummy.csv"
    df.to_csv(csv_path, index=False)
    cfg = DummyConfig(csv_path=str(csv_path))
    loaded = cfg.data_frame
    assert set(loaded.columns) == {"a", "b"}
    assert loaded.iloc[0]["a"] == 1
    assert loaded.iloc[0]["b"] == 2
