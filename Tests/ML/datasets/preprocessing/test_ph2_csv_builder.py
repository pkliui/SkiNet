import argparse
from pathlib import Path

import pandas as pd
import pytest

from SkiNet.ML.datasets.preprocessing.ph2_csv_builder import PH2BaseCSVBuilder, PH2LocalCSVBuilder
from SkiNet.Utils.csv_headers import (DATAPATH_HEADER, DATATYPE_HEADER, PH2_COLORS_HEADER, PH2_COLORS_LIST_HEADER,
                                      PH2_NAME_HEADER, SAMPLEID_HEADER)
from SkiNet.Utils.project_paths import PH2_TXT_NAME

# -------------- fixtures and dummy class ----------------

class TestLocalPH2Base(PH2BaseCSVBuilder):
    def __init__(self) -> None:
        self._local_data_root = "/dummy/local/root"

    @property
    def data_root(self) -> str:
        return self._local_data_root

    def datapath_func(self, path: str) -> str:
        return path  # or any dummy logic

    @property
    def external_txt_name(self) -> str:
        return PH2_TXT_NAME

    @property
    def external_txt_path(self) -> str:
        return f"{self.data_root}/{self.external_txt_name}"

    @property
    def image_pattern(self) -> str:
        return "**/*.bmp"

    @property
    def mask_pattern(self) -> str:
        return "**/*_lesion.bmp"

    def parse_external_txt(self) -> pd.DataFrame:
        return pd.DataFrame()

    def create_metadata_csv(self) -> None:
        pass

    def merge_ph2_data(self, basic_df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
        return super().merge_ph2_data(basic_df, external_df)

    def sampleid_func(self, path: str) -> str:
        return super().sampleid_func(path)


@pytest.fixture
def tmp_ph2_txt(tmp_path: Path) -> Path:
    """
    Create a temporary PH2 dataset text file.
    """
    txt = tmp_path / PH2_TXT_NAME
    txt.write_text(
        "|| Name || Age || Diagnosis || a|b|c ||\n"
        "|| sample1 || 45 || benign || 4|5|6 ||\n"
        "|| sample2 || 50 || malignant || 7|8|9 ||\n"
    )
    return txt


# -----------Tests for PH2BaseCSVBuilder----------------

def test_parse_txt_lines_basic() -> None:
    """
    Test parsing of basic text lines into a DataFrame.
    _parse_txt_lines expects lines in a specific format, where the first line contains headers separated by "||" and subsequent lines contain data.
    """
    lines = [
        "|| Name || Age || Diagnosis || a|b|c ||",
        "|| sample1 || 45 || benign || 4|5|6 ||",
        "|| sample2 || 50 || malignant || 7|8|10 ||"
    ]
    df = PH2BaseCSVBuilder._parse_txt_lines(lines)
    expected = pd.DataFrame([
        {"Name": "sample1", "Age": "45", "Diagnosis": "benign", "a": "4", "b": "5", "c": "6"},
        {"Name": "sample2", "Age": "50", "Diagnosis": "malignant", "a": "7", "b": "8", "c": "10"},
    ])

    # Reindex columns to ensure order matches
    expected = expected[["Name", "Age", "Diagnosis", "a", "b", "c"]]
    df = df[["Name", "Age", "Diagnosis", "a", "b", "c"]]

    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected.reset_index(drop=True))


def test_merge_ph2_data_merges_and_processes_colors() -> None:
    """
    Test merging of basic and external DataFrames, ensuring COLORS_HEADER is processed into COLORS_LIST_HEADER and merged correctly,
    and that COLORS_HEADER is removed from the final DataFrame.
    """
    base = TestLocalPH2Base()
    basic_df = pd.DataFrame([
        {SAMPLEID_HEADER: "sample1", DATAPATH_HEADER: "foo/bar1", DATATYPE_HEADER: "image"},
        {SAMPLEID_HEADER: "sample2", DATAPATH_HEADER: "foo/bar2", DATATYPE_HEADER: "mask"}
    ])
    external_df = pd.DataFrame([
        {PH2_NAME_HEADER: "sample1", PH2_COLORS_HEADER: "1 2 3"},
        {PH2_NAME_HEADER: "sample2", PH2_COLORS_HEADER: "4 5"}
    ])
    merged = base.merge_ph2_data(basic_df, external_df)
    assert PH2_COLORS_LIST_HEADER in merged.columns
    assert merged.loc[merged[SAMPLEID_HEADER] == "sample1", PH2_COLORS_LIST_HEADER].iloc[0] == [1, 2, 3]
    assert merged.loc[merged[SAMPLEID_HEADER] == "sample2", PH2_COLORS_LIST_HEADER].iloc[0] == [4, 5]
    assert PH2_COLORS_HEADER not in merged.columns

def test_sampleid_func_extracts_sampleid() -> None:
    """
    Test extraction of sample IDs from file paths.
    """
    base = TestLocalPH2Base()
    path = "/root/sample1/sample1_Dermoscopic_Image/sample1.bmp"
    assert base.sampleid_func(path) == "sample1"
    path2 = "sample2/sample2_lesion/sample2_lesion.bmp"
    assert base.sampleid_func(path2) == "sample2"

def test_ph2localcsvbuilder_parse_external_txt(tmp_path: Path, tmp_ph2_txt: Path) -> None:
    """
    Test parsing of the external PH2 dataset text file.
    """
    args = argparse.Namespace(local_data_root=str(tmp_path))
    builder = PH2LocalCSVBuilder(args)

    df = builder.parse_external_txt()
    expected = pd.DataFrame([
        {"Name": "sample1", "Age": "45", "Diagnosis": "benign", "a": "4", "b": "5", "c": "6"},
        {"Name": "sample2", "Age": "50", "Diagnosis": "malignant", "a": "7", "b": "8", "c": "9"},
    ])
    # Reindex columns to ensure order matches
    expected = expected[["Name", "Age", "Diagnosis", "a", "b", "c"]]
    df = df[["Name", "Age", "Diagnosis", "a", "b", "c"]]

    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected.reset_index(drop=True))


def test_ph2localcsvbuilder_parse_external_txt_file_not_found(tmp_path: Path) -> None:
    """
    Test handling of missing external TXT file in PH2LocalCSVBuilder.
    """
    args = argparse.Namespace(local_data_root=str(tmp_path))
    builder = PH2LocalCSVBuilder(args)
    # Remove the TXT file if it exists
    txt_path = tmp_path / PH2_TXT_NAME
    if txt_path.exists():
        txt_path.unlink()
    with pytest.raises(FileNotFoundError):
        builder.parse_external_txt()


def test_ph2localcsvbuilder_parse_external_txt_empty(tmp_path: Path) -> None:
    """
    Test handling of empty external TXT file in PH2LocalCSVBuilder.
    """
    txt_path = tmp_path / PH2_TXT_NAME
    txt_path.write_text("")
    args = argparse.Namespace(local_data_root=str(tmp_path))
    builder = PH2LocalCSVBuilder(args)
    with pytest.raises(ValueError, match="is empty or contains no valid lines"):
        builder.parse_external_txt()

def test_ph2localcsvbuilder_parse_external_txt_incorrect_format(tmp_path: Path) -> None:
    """
    Test handling of a malformed external TXT file in PH2LocalCSVBuilder.
    The file is missing the expected '||' separators or has inconsistent columns.
    """
    # Malformed: missing '||' at the start, or wrong number of columns
    txt_path = tmp_path / PH2_TXT_NAME
    txt_path.write_text(
        "Name | Age | Diagnosis | a|b|c\n"
        "sample1 | 45 | benign | 4|5|6\n"
        "sample2 | 50 | malignant | 7|8|9\n"
    )
    args = argparse.Namespace(local_data_root=str(tmp_path))
    builder = PH2LocalCSVBuilder(args)
    with pytest.raises(ValueError, match="is empty or contains no valid lines"):
        builder.parse_external_txt()


def test_merge_ph2_data_missing_or_empty_colors() -> None:
    """
    Test merging when the Colors column is missing or empty in the external DataFrame.
    """
    base = TestLocalPH2Base()
    # No Colors column for sample1, empty for sample2
    basic_df = pd.DataFrame([
        {SAMPLEID_HEADER: "sample1", DATAPATH_HEADER: "foo/bar1", DATATYPE_HEADER: "image"},
        {SAMPLEID_HEADER: "sample2", DATAPATH_HEADER: "foo/bar2", DATATYPE_HEADER: "mask"}
    ])
    external_df = pd.DataFrame([
        {PH2_NAME_HEADER: "sample1"},  # No Colors column
        {PH2_NAME_HEADER: "sample2", PH2_COLORS_HEADER: ""}  # Empty Colors
    ])
    merged = base.merge_ph2_data(basic_df, external_df)
    # If Colors column is missing, PH2_COLORS_LIST_HEADER should still be present and be empty list
    assert PH2_COLORS_LIST_HEADER in merged.columns
    assert merged.loc[merged[SAMPLEID_HEADER] == "sample1", PH2_COLORS_LIST_HEADER].iloc[0] == []
    assert merged.loc[merged[SAMPLEID_HEADER] == "sample2", PH2_COLORS_LIST_HEADER].iloc[0] == []
