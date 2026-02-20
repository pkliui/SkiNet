import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from azureml.fsspec import AzureMachineLearningFileSystem
from numpy.typing import NDArray

from SkiNet.Azure.azure_setup import AzureSetup
from SkiNet.ML.configs.datasets.dataset_keys import AzureDatasetKey
from SkiNet.ML.utils.data_utils import convert_to_numpy_bytes, filter_missing_images_and_masks
from SkiNet.Utils.csv_headers import DATAPATH_HEADER, DATATYPE_HEADER, SAMPLEID_HEADER
from SkiNet.Utils.loggers import file_logging, stdout_logging
from SkiNet.Utils.project_paths import PH2_CSV_NAME, PH2_TXT_NAME

"""
This script prepares the PH2 dataset for using with SkiNet by creating a metadata CSV
that maps sample IDs to their corresponding image and mask file paths.

The resulting format of the CSV is expected by the SkiNet dataset classes.
"""

def get_ph2_data_paths(data_root: Union[Path, AzureMachineLearningFileSystem]) -> Tuple[NDArray[np.bytes_], NDArray[np.bytes_]]:
    """
    Get paths to images and masks of a PH2 dataset. Paths whose images and masks do not have the same dimensions are filtered out.

    PH2 dataset copyright: Teresa Mendonça, Pedro M. Ferreira, Jorge Marques, Andre R. S. Marcal, Jorge Rozeira.
    PH² - A dermoscopic image database for research and benchmarking,
    35th International Conference of the IEEE Engineering in Medicine and Biology Society, July 3-7, 2013, Osaka, Japan.

    :param data_root: if provided as a Path, it is a local directory that contains folders with samples of data uniquely identifiable by their ID,
        otherwise it is an Azure Machine Learning filesystem instance referencing a specific location on Azure data storage.
    :return: A tuple of two numpy arrays of dtype np.bytes_ that contains full paths to images and masks.
        This data type is required by the Dataset class to fix the copy-on-write problem that results in an increased memory usage
        https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading

    The file structure is expected to be as following (as per original PH2 file structure):
        * root_dir
            * sample1
                * sample1_Dermoscopic_Image
                    * sample1.bmp
                * sample1_lesion
                *    sample1_lesion.bmp
            * sample2
                * sample2_Dermoscopic_Image
                    * sample1.bmp
                * sample2_lesion
                    * sample2_lesion.bmp

    Note that only ONE image per folder is expected,
    e.g. one image in sample1_Dermoscopic_Image folder, one mask in sample1_lesion folder
    """
    if isinstance(data_root, Path):
        if not data_root.is_dir():
            raise ValueError(f"The specified data_root does not exist: {data_root}")
        image_paths = list(data_root.rglob("*_Dermoscopic_Image/*.bmp"))
        mask_paths = list(data_root.rglob("*_lesion/*.bmp"))
    elif isinstance(data_root, AzureMachineLearningFileSystem):
        try:
            image_paths = list(data_root.glob("**_Dermoscopic_Image/**.bmp"))
            mask_paths = list(data_root.glob("**_lesion/**.bmp"))
        except Exception as e:
            raise ValueError(f"Error occurred while accessing Azure file system: {e}")

    image_paths, mask_paths = filter_missing_images_and_masks(image_paths, mask_paths)

    # Convert lists to NumPy arrays of dtype np.bytes_ to fix copy-on-write problem that results in an increased memory usage
    # https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
    image_paths_array, mask_paths_array = convert_to_numpy_bytes(image_paths, mask_paths)
    return image_paths_array, mask_paths_array


def create_ph2_metadata(local_data_source: Optional[Path] = None,
                        azure_data: bool = False) -> pd.DataFrame:
    """
    Create a CSV file for the PH2 dataset

    :param local_data_source: If azure_data is False, this is a local path to the PH2 dataset root folder.
        If azure_data is True, this argument is ignored and an error is raised if provided.
    :param azure_data: If True, handles data from AzureMachineLearningFileSystem instance.

    :return: A DataFrame containing the PH2 dataset metadata containing the following columns:
        - sampleid: unique identifier for each sample, extracted from the folder structure of the dataset
        - datapath: the path to the image or mask file, relative to the local_data_source if azure_data is False,
        or the full path on Azure if azure_data is True
        - datatype: the type of the data, either "image" or "mask"
    """

    # set up logging
    stdout_logging(logging.DEBUG)
    file_logging()

    # get paths to images and masks
    if azure_data:
        logging.getLogger(__name__).info("Setting up metadata file in Azure")
        assert local_data_source is None, "Do not pass local_data_source when using Azure."

        # Authenticate and get the Azure file system
        AzureSetup.service_principal_authentication()
        fs = AzureSetup.get_azureml_filesystem(AzureDatasetKey.PH2_DATASET_KEY.value)
        logging.getLogger(__name__).info(f"Azure data fs: {fs.ls()}")

        image_paths, mask_paths = get_ph2_data_paths(fs)
    else:
        logging.getLogger(__name__).info("Setting up metadata file in local file system")
        assert isinstance(local_data_source, Path), "When using local data, local_data_source must be a Path."

        image_paths, mask_paths = get_ph2_data_paths(local_data_source)

    # Create metadata entries
    data = []
    for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        # Decode bytes to string if necessary
        img_path_str = img_path.decode("utf-8") if isinstance(img_path, bytes) else str(img_path)
        mask_path_str = mask_path.decode("utf-8") if isinstance(mask_path, bytes) else str(mask_path)

        # Extract sample ID
        sample_id = Path(img_path_str).parent.parent.name

        # Make paths relative to local_data_source
        if not azure_data:
            assert isinstance(local_data_source, Path), "When using local data, local_data_source must be a Path."
            img_rel_path = str(Path(img_path_str).relative_to(local_data_source))
            mask_rel_path = str(Path(mask_path_str).relative_to(local_data_source))
        else:
            img_rel_path = img_path_str
            mask_rel_path = mask_path_str

        # Add entries for image and mask
        data.append({SAMPLEID_HEADER: sample_id, DATAPATH_HEADER: img_rel_path, DATATYPE_HEADER: "image"})
        data.append({SAMPLEID_HEADER: sample_id, DATAPATH_HEADER: mask_rel_path, DATATYPE_HEADER: "mask"})

    df = pd.DataFrame(data)
    return df


def parse_ph2_txt(txt_path: str, feat_col_idx: int = 3) -> pd.DataFrame:
    """
    Parse the PH2 dataset TXT file.
    The TXT file has a specific format where the first line contains headers, and subsequent lines contain data.

    :param txt_path: The path to the TXT file.
    :param feat_col_idx: The index of the features column (default is 3).

    :return: A DataFrame containing the parsed PH2 dataset metadata.
    """

    FEAT_COL_IDX: int = feat_col_idx  # The index of the features column in the TXT file where the features are separated by '|'

    def _split_lines(line: str) -> list[str]:
        # Remove trailing and leading pipes and then split on double pipes
        line_parts = [ll.strip() for ll in line.strip('|').split('||')]
        # The features part needs to be split further
        features = [f.strip() for f in line_parts[FEAT_COL_IDX].split('|')]
        # Processed line with features split into separate columns
        processed_line = line_parts[:FEAT_COL_IDX] + features + line_parts[FEAT_COL_IDX + 1:]
        assert len(processed_line) == len(line_parts) - 1 + len(features), "Line length mismatch after splitting features"
        return processed_line

    # Only keep lines that start with '||', i.e. the header and data lines
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip().startswith('||')]

    header_line = lines[0]
    headers = _split_lines(header_line)

    data = []
    for line in lines[1:]:
        data_row = _split_lines(line)
        data.append(data_row)

    df = pd.DataFrame(data, columns=headers)
    return df


def merge_ph2_data(df_csv: pd.DataFrame, df_txt: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the PH2 dataset CSV and TXT metadata.

    :param df_csv: DataFrame containing the CSV metadata with columns [sampleid, datapath, datatype].
    :param df_txt: DataFrame containing the TXT metadata with a 'Name' column that corresponds to 'sampleid' in the CSV, and a 'Colors' column with features.

    :return: A merged DataFrame containing the combined metadata from both sources, with the 'Colors' column processed into a list of integers.
    """
    df_merged = df_csv.merge(df_txt, left_on="sampleid", right_on="Name", how="left")

    if 'Colors' in df_merged.columns:
        df_merged['Colors_list'] = df_merged['Colors'].apply(lambda x: [int(i) for i in str(x).split()] if pd.notnull(x) else [])
        df_merged = df_merged.drop(columns=['Colors'])

    return df_merged


def save_dataframe_to_csv(df: pd.DataFrame, output_csv_path: Union[str, Path], azure_data: bool, dataset_name: Optional[str] = None) -> None:
    """
    Save a DataFrame to a CSV file. If azure_data is True, the CSV file is first saved locally and then uploaded to Azure Blob Storage.

    :param df: The DataFrame to save.
    :param output_csv_path: Local path where the CSV file will be saved.
    :param azure_data: If True, the CSV file will be uploaded to Azure Blob Storage after being saved locally.
    :param dataset_name: The name of the dataset used in code, i.e., a key from the PATH_ON_DATASTORE dictionary in your Azure YAML config.
        This is required if azure_data is True, as it is used to determine the location on Azure where the CSV file will be uploaded.
    """
    try:
        df.to_csv(output_csv_path, index=False)
        logging.getLogger(__name__).info(f"CSV file created at {output_csv_path} with {len(df)} entries.")
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save CSV file at {output_csv_path}: {e}")
        raise

    if azure_data:
        if dataset_name is None:
            raise ValueError("dataset_name must be provided when uploading to Azure.")
        upload_csv_to_blob(str(output_csv_path), dataset_name)
        logging.getLogger(__name__).info(
            f"CSV file with the local path {output_csv_path} was saved on Azure filesystem {dataset_name} with {len(df)} entries.")
        try:
            os.remove(output_csv_path)
            logging.getLogger(__name__).info(f"Temporary local CSV file {output_csv_path} has been removed.")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not remove temporary file {output_csv_path}: {e}")


def upload_csv_to_blob(local_csv_path: str, dataset_name: str) -> None:
    """
    Upload a CSV file to Azure Blob Storage using AzureSetup authentication.

    :param local_csv_path: Local path to the CSV file to upload.
    :param dataset_name: The name of the dataset used in code, i.e., a key from the PATH_ON_DATASTORE dictionary in your Azure YAML config.
    """
    AzureSetup.service_principal_authentication()
    fs = AzureSetup.get_azureml_filesystem(dataset_name)
    # get the path to dataset folder on Azure
    _, path_on_azure = AzureSetup.get_azure_uri(dataset_name)

    # recursive must be set to False to upload a file
    fs.upload(lpath=local_csv_path, rpath=path_on_azure, recursive=False, **{'overwrite': 'MERGE_WITH_OVERWRITE'})


def main_create_ph2_metadata_from_args(args: argparse.Namespace) -> None:

    if args.azure_data and args.local_data_source is not None:
        raise ValueError("Do not provide --local-data-source when using --azure-data.")

    if args.azure_data:
        output_csv_path = PH2_CSV_NAME  # This is the name of the CSV file that will be saved on Azure
        df = create_ph2_metadata(azure_data=args.azure_data)
        df_txt = parse_ph2_txt(PH2_TXT_NAME)
        df_merged = merge_ph2_data(df_csv=df, df_txt=df_txt)
        # Ensure dataset_name is not None
        save_dataframe_to_csv(df=df_merged, output_csv_path=output_csv_path, azure_data=args.azure_data, dataset_name=AzureDatasetKey.PH2_DATASET_KEY.value)

    else:
        output_csv_path = str(Path(args.local_data_source) / PH2_CSV_NAME)
        df = create_ph2_metadata(local_data_source=Path(args.local_data_source),
                                 azure_data=args.azure_data)
        df_txt = parse_ph2_txt(str(Path(args.local_data_source) / PH2_TXT_NAME))
        df_merged = merge_ph2_data(df_csv=df, df_txt=df_txt)
        save_dataframe_to_csv(df=df_merged, output_csv_path=output_csv_path, azure_data=args.azure_data)


if __name__ == "__main__":
    """
    Entry point for creating PH2 metadata.

    Example for local data:
    ```
    create_ph2_metadata(local_data_source=Path("path/to/local/data"),
                        output_csv_path=Path("path/to/output.csv"),
                        azure_data=False)
    ```

    Example for Azure data:
    ```
    create_ph2_metadata(azure_data=True)
    ```
    """
    parser = argparse.ArgumentParser(description="Create PH2 metadata CSV for SkiNet.")
    parser.add_argument("--azure-data",
                        action='store_true',
                        required=False,
                        help="If True, local_data_source is an AzureMachineLearningFileSystem instance, otherwise it is a local Path.")
    parser.add_argument("--local-data-source",
                        type=str,
                        required=False,
                        help="Only used when no --azure-data  flag, as this is the local path to the PH2 dataset root folder")
    args = parser.parse_args()
    main_create_ph2_metadata_from_args(args)
