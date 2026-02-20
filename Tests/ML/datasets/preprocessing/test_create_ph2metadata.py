from pathlib import Path

import pandas as pd
import pytest

from SkiNet.ML.datasets.preprocessing.create_ph2_metadata import create_ph2_metadata, save_dataframe_to_csv
from SkiNet.Utils.csv_headers import DATAPATH_HEADER, DATATYPE_HEADER, SAMPLEID_HEADER
from SkiNet.Utils.project_paths import PH2_CSV_NAME


@pytest.mark.parametrize(
    "sample_structure,expected_subjects, expected_file_paths, expected_channels",
    [
        (
            {
                "sample1/sample1_Dermoscopic_Image/sample1.bmp": b"image",
                "sample1/sample1_lesion/sample1_lesion.bmp": b"mask"
            },
            ["sample1"], ["sample1/sample1_Dermoscopic_Image/sample1.bmp", "sample1/sample1_lesion/sample1_lesion.bmp"], ["image", "mask"]
        ),
        (
            {
                "sample1/sample1_Dermoscopic_Image/sample1.bmp": b"image",
                "sample1/sample1_lesion/sample1_lesion.bmp": b"mask",
                "sample2/sample2_Dermoscopic_Image/sample2.bmp": b"image",
                "sample2/sample2_lesion/sample2_lesion.bmp": b"mask"
            },
            ["sample1", "sample2"], ["sample1/sample1_Dermoscopic_Image/sample1.bmp", "sample1/sample1_lesion/sample1_lesion.bmp",
                                     "sample2/sample2_Dermoscopic_Image/sample2.bmp", "sample2/sample2_lesion/sample2_lesion.bmp"], ["image", "mask"]
        ),
    ]
)
def test_create_ph2_metadata_local(tmp_path: Path,
                                   sample_structure: dict[str, bytes],
                                   expected_subjects: list[str],
                                   expected_file_paths: list[str],
                                   expected_channels: list[str]) -> None:
    """
    Test create_ph2_metadata function with local file system.
    """
    # loop through the sample_structure and save images and masks as per their given locations
    for rel_path, content in sample_structure.items():
        file_path = tmp_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)

    output_csv = tmp_path / PH2_CSV_NAME
    df = create_ph2_metadata(local_data_source=tmp_path, azure_data=False)
    save_dataframe_to_csv(df=df, output_csv_path=output_csv, azure_data=False)

    # read the csv
    df = pd.read_csv(output_csv)

    # Check subject and channel columns
    assert set(df[SAMPLEID_HEADER]) == set(expected_subjects)
    assert set(df[DATATYPE_HEADER]) == set(expected_channels)

    # Compare actual filePath values to expected
    actual_paths = set(df[DATAPATH_HEADER])
    assert actual_paths == set(expected_file_paths), f"Actual paths: {actual_paths}, Expected: {expected_file_paths}"
    for fp in df[DATAPATH_HEADER]:
        assert not str(fp).startswith(str(tmp_path))
