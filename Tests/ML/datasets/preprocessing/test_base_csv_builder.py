import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from SkiNet.ML.datasets.preprocessing.base_csv_builder import AzureCSVBuilder, LocalCSVBuilder
from SkiNet.Utils.csv_headers import DATAPATH_HEADER, DATATYPE_HEADER, SAMPLEID_HEADER

# -------------- fixtures  for LocalCSVBuilder -----------------

@pytest.fixture
def tmp_local_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with a structure that mimics the expected local dataset layout."""
    for sample, img, mask in [("sample1", "image1.jpg", "mask1.png"), ("sample2", "image2.jpg", "mask2.png")]:
        img_dir = tmp_path / sample
        img_dir.mkdir()
        (img_dir / img).write_text("fake image")
        (img_dir / mask).write_text("fake mask")
    return tmp_path

@pytest.fixture
def local_builder(tmp_local_dir: Path) -> LocalCSVBuilder:
    """ Create a LocalCSVBuilder instance with dummy arguments for testing.
    The builder will use the temporary directory created by tmp_local_dir as its data root."""
    class TestLocalBuilder(LocalCSVBuilder):
        @property
        def image_pattern(self) -> str:
            """Pattern to match image files as defined in tmp_local_dir structure"""
            return "**/*.jpg"

        @property
        def mask_pattern(self) -> str:
            """Pattern to match mask files as defined in tmp_local_dir structure"""
            return "**/*.png"

        @property
        def output_csv_name(self) -> str:
            return "test.csv"

        def sampleid_func(self, path_str: str) -> str:
            # we are one level down from the sample id in tmp_local_dir, so parent should give us the sample id
            return Path(path_str).parent.name

        def create_metadata_csv(self) -> None:
            pass

    # ensure that TestLocalBuilder receives a Namespace object with a .local_data_root attribute
    args = argparse.Namespace(local_data_root=str(tmp_local_dir))
    return TestLocalBuilder(args)


# -------------- tests for LocalCSVBuilder -----------------

@pytest.mark.parametrize("expected_type", [Path])
def test_local_data_root(local_builder: LocalCSVBuilder, tmp_local_dir: Path, expected_type: type) -> None:
    """Test that data_root instance variable returns the correct type and value."""
    assert isinstance(local_builder.data_root, expected_type)
    assert local_builder.data_root == tmp_local_dir


@pytest.mark.parametrize("path_str,expected", [
    ("sample1/image1.jpg", "sample1"),
    ("sample2/mask2.png", "sample2"),
])
def test_sampleid_func(local_builder: LocalCSVBuilder, tmp_local_dir: Path, path_str: str, expected: str) -> None:
    """Test sampleid_func returns the correct sample id from a path."""
    full_path = str(tmp_local_dir / path_str)
    assert local_builder.sampleid_func(full_path) == expected


@pytest.mark.parametrize("sub_path", [
    "sampleA/imageA.jpg",
    "sampleA/maskA.png",
])
def test_local_datapath_func(local_builder: LocalCSVBuilder, tmp_local_dir: Path, sub_path: str) -> None:
    """
    Test LocalCSVBuilder.datapath_func to ensure it correctly computes the relative path from the full path to the data root.
    """
    # construct full path to data
    full_path = str(tmp_local_dir / sub_path)
    # get the path relative to the data root, i.e. tmp_local_dir
    rel = local_builder.datapath_func(full_path)
    assert rel == sub_path

@pytest.mark.parametrize("samples", [
    [("sample1", "image1.jpg", "mask1.png"), ("sample2", "image2.jpg", "mask2.png")]
])
def test_get_data_paths_and_image_and_mask_paths(local_builder: LocalCSVBuilder, tmp_local_dir: Path, samples: list[tuple[str, str, str]]) -> None:
    """
    Test get_data_paths and get_image_and_mask_paths return correct arrays of bytes.
    """
    image_paths, mask_paths = local_builder.get_data_paths()
    # Should find 2 images and 2 masks as per the structure created by tmp_local_dir fixture
    assert isinstance(image_paths, np.ndarray)
    assert isinstance(mask_paths, np.ndarray)
    assert image_paths.dtype.kind == 'S'
    assert mask_paths.dtype.kind == 'S'

    # Build expected values directly
    expected_images = [
        str(tmp_local_dir / sample / img).encode()
        for sample, img, _ in samples
    ]
    expected_masks = [
        str(tmp_local_dir / sample / mask).encode()
        for sample, _, mask in samples
    ]

    # Ensure the correct images and masks are found using get_data_paths()
    assert sorted(image_paths.tolist()) == sorted(expected_images)
    assert sorted(mask_paths.tolist()) == sorted(expected_masks)

    # Test get_image_and_mask_paths directly
    img_list_from_get_image_and_mask_paths, mask_list_from_get_image_and_mask_paths = local_builder.get_image_and_mask_paths(
        local_builder.data_root,
        local_builder.image_pattern,
        local_builder.mask_pattern
    )
    # Ensure the same results are returned by both methods
    assert np.array_equal(sorted(image_paths), sorted(img_list_from_get_image_and_mask_paths))
    assert np.array_equal(sorted(mask_paths), sorted(mask_list_from_get_image_and_mask_paths))


@pytest.mark.parametrize("sampleid, img,mask", [
    ("sample1", "sample1/image1.jpg", "sample1/mask1.png"),
    ("sample2", "sample2/image2.jpg", "sample2/mask2.png"),
])
def test_create_dataframe_with_paths_and_types(local_builder: LocalCSVBuilder, tmp_local_dir: Path, sampleid: str, img: str, mask: str) -> None:
    """
    Test LocalCSVBuilder.create_dataframe_with_paths_and_types to ensure it correctly creates a DataFrame
    with the expected columns and values based on the provided image and mask paths.

    Note: img, mask params match the structure of the temporary directory created by tmp_local_dir fixture.
    Also because create_dataframe_with_paths_and_types accepts only specific paths per test parameterization,
    the DataFrame will  contain all entries for these specific paths
    (two entries for a pair of image and mask in this case), rather than all entries in the directory.
    """
    # use full paths for image and mask based on the temporary directory structure
    img_path = str(tmp_local_dir / img)
    mask_path = str(tmp_local_dir / mask)

    df = local_builder.create_dataframe_with_paths_and_types(
        image_paths=np.array([img_path.encode()], dtype=np.bytes_),
        mask_paths=np.array([mask_path.encode()], dtype=np.bytes_),
        sampleid_func=lambda x: Path(x).parent.name,  # use the parent directory as the sample ID
        datapath_func=local_builder.datapath_func  # use the builder's datapath_func to get relative paths in the DataFrame
    )
    assert set(df.columns) == {SAMPLEID_HEADER, DATAPATH_HEADER, DATATYPE_HEADER}
    assert len(df) == 2  # only 1 pair per test case

    assert sampleid in df[SAMPLEID_HEADER].values
    assert "image" in df[DATATYPE_HEADER].values
    assert "mask" in df[DATATYPE_HEADER].values

    expected_paths = [img, mask]
    for expected in expected_paths:
        assert expected in df[DATAPATH_HEADER].values

@pytest.mark.parametrize("sampleid, img,mask", [
    ("sample1", "sample1/image1.jpg", "sample1/mask1.png"),
    ("sample2", "sample2/image2.jpg", "sample2/mask2.png"),
])
def test_create_basic_metadata(local_builder: LocalCSVBuilder, sampleid: str, img: str, mask: str) -> None:
    """
    Test create_basic_metadata returns a DataFrame with correct structure and values

    Namely we expect that create_basic_metadata() scans the whole data root and returns all entries
    via self.get_data_paths();
    as a result, the DataFrame should contain all samples as per the directory structure (4 entries in this case,
    two pairs of image and mask)
    """
    df = local_builder.create_basic_metadata()

    assert set(df.columns) == {SAMPLEID_HEADER, DATAPATH_HEADER, DATATYPE_HEADER}
    assert len(df) == 4  # all pairs in the directory per test case, i.e. 4 in this case

    assert sampleid in df[SAMPLEID_HEADER].values
    assert "image" in df[DATATYPE_HEADER].values
    assert "mask" in df[DATATYPE_HEADER].values

    expected_paths = [img, mask]
    for expected in expected_paths:
        assert expected in df[DATAPATH_HEADER].values


def test_save_dataframe_to_csv(tmp_path: Path, local_builder: LocalCSVBuilder) -> None:
    """
    Test LocalCSVBuilder.save_dataframe_to_csv to ensure it correctly saves the DataFrame to a CSV file.
    """
    df = pd.DataFrame([
        {SAMPLEID_HEADER: "id1", DATAPATH_HEADER: "foo/bar1", DATATYPE_HEADER: "image"},
        {SAMPLEID_HEADER: "id2", DATAPATH_HEADER: "foo/bar2", DATATYPE_HEADER: "mask"}
    ])
    csv_path = tmp_path / "out.csv"
    local_builder.save_dataframe_to_csv(df, csv_path)
    loaded = pd.read_csv(csv_path)

    assert loaded.shape == (2, 3)
    assert set(loaded.columns) == {SAMPLEID_HEADER, DATAPATH_HEADER, DATATYPE_HEADER}

    # assert values in loaded csv match the original DataFrame
    for i, row in df.iterrows():
        for key, value in row.items():
            assert loaded.iloc[i][key] == value


# -------------- tests for AzureCSVBuilder -----------------

@pytest.fixture
def azure_builder() -> AzureCSVBuilder:
    """
    Create an AzureCSVBuilder for testing purposes.
    """
    class TestAzureBuilder(AzureCSVBuilder):
        def __init__(self, dataset_name: str = "dummy", data_root_on_azure: str = "PH2DATA/") -> None:
            self.fs = None
            self._data_root_on_azure = data_root_on_azure
            self._dataset_name = dataset_name

        @property
        def data_root_on_azure(self) -> str:
            return self._data_root_on_azure

        @property
        def image_pattern(self) -> str:
            return "**/*.jpg"

        @property
        def mask_pattern(self) -> str:
            return "**/*.png"

        @property
        def output_csv_name(self) -> str:
            return "test.csv"

        @property
        def data_root(self) -> str:
            # NB: return the Azure data root path _data_root_on_azure only here in the test, because we are not using a file system of Azure
            # in the code, data_root returns AzureMachineLearningFileSystem instance
            return self._data_root_on_azure

        def sampleid_func(self, path_str: str) -> str:
            # just as for LocalCSVBuilder, we can use the stem of the file name as the sample ID for testing purposes
            return Path(path_str).parent.name

        def create_metadata_csv(self) -> None:
            pass

    return TestAzureBuilder()

def test_azure_builder_properties(azure_builder: AzureCSVBuilder) -> None:
    """
    Test properties of the AzureCSVBuilder.
    """
    assert azure_builder.data_root == "PH2DATA/"
    assert isinstance(azure_builder.image_pattern, str)
    assert isinstance(azure_builder.mask_pattern, str)
    assert isinstance(azure_builder.output_csv_name, str)
    assert callable(azure_builder.sampleid_func)

@pytest.mark.parametrize("data_root_on_azure,path_str,expected", [
    ("PH2DATA/", "PH2DATA/sample1/image1.jpg", "sample1/image1.jpg"),
    ("PH2DATA_V1/", "PH2DATA_V1/sample1/image1.jpg", "sample1/image1.jpg"),  # change the default root
    ("PH2DATA_V2/", "PH2DATA_V2/sample1/image1.jpg", "sample1/image1.jpg"),
    ("PH2DATA_V3/", "sample1/image1.jpg", "sample1/image1.jpg"),  # no root in the path_str
])
def test_azure_datapath_func(azure_builder: AzureCSVBuilder, data_root_on_azure: str, path_str: str, expected: str) -> None:
    """
    Check that AzureCSVBuilder.datapath_func correctly computes the relative path from the full path to the Azure data root.
    """
    # Reconfigure the fixture for each test case
    azure_builder._data_root_on_azure = data_root_on_azure
    assert azure_builder.datapath_func(path_str) == expected


def test_azure_sampleid_func(azure_builder: AzureCSVBuilder) -> None:
    """
    Test that AzureCSVBuilder.sampleid_func correctly extracts the sample ID from a given path string.
    """
    assert azure_builder.sampleid_func("sample1/image1.jpg") == "sample1"
    assert azure_builder.sampleid_func("sample2/mask2.png") == "sample2"
