import logging
import os
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from azureml.fsspec import AzureMachineLearningFileSystem
from numpy.typing import NDArray

from SkiNet.Azure.azure_setup import AzureSetup
from SkiNet.ML.utils.data_utils import convert_to_numpy_bytes, filter_missing_images_and_masks
from SkiNet.Utils.loggers import file_logging, stdout_logging

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


def create_ph2_metadata(data_source: Union[Path, str],
                        output_csv_path: Union[str, Path],
                        azure_data: bool = False) -> None:
    """
    Create a CSV file for the PH2 dataset

    :param data_source: If azure_data is False, this is a local path to the PH2 dataset root folder.
        If azure_data is True, this is a name of the dataset used in code, i.e., a key from the PATH_ON_DATASTORE dictionary in your Azure YAML config.
    :param output_csv_path: Local path where the CSV file will be saved.
        If azure_data is True, this is the temporary local path where the CSV file will be saved before uploading to Azure.
    :param azure_data: If True, data_source is an AzureMachineLearningFileSystem instance.
    """

    # set up logging
    stdout_logging(logging.DEBUG)
    file_logging()

    # get paths to images and masks
    if azure_data:
        logging.getLogger(__name__).info("Setting up metadata file in Azure")
        assert isinstance(data_source, str), "When using Azure data, data_source must be a dataset name (str)."

        # Authenticate and get the Azure file system
        AzureSetup.service_principal_authentication()
        fs = AzureSetup.get_azureml_filesystem(data_source)
        logging.getLogger(__name__).info(f"Azure data fs: {fs.ls()}")

        image_paths, mask_paths = get_ph2_data_paths(fs)
    else:
        logging.getLogger(__name__).info("Setting up metadata file in local file system")
        assert isinstance(data_source, Path), "When using local data, data_source must be a Path."

        image_paths, mask_paths = get_ph2_data_paths(data_source)

    # Create metadata entries
    data = []
    for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        # Decode bytes to string if necessary
        img_path_str = img_path.decode("utf-8") if isinstance(img_path, bytes) else str(img_path)
        mask_path_str = mask_path.decode("utf-8") if isinstance(mask_path, bytes) else str(mask_path)

        # Extract sample ID
        sample_id = Path(img_path_str).parent.parent.name

        # Make paths relative to data_source
        if not azure_data:
            img_rel_path = str(Path(img_path_str).relative_to(data_source))
            mask_rel_path = str(Path(mask_path_str).relative_to(data_source))
        else:
            img_rel_path = img_path_str
            mask_rel_path = mask_path_str

        # Add entries for image and mask
        data.append({"sampleid": sample_id, "datapath": img_rel_path, "datatype": "image"})
        data.append({"sampleid": sample_id, "datapath": mask_rel_path, "datatype": "mask"})

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    try:
        df.to_csv(output_csv_path, index=False)
        logging.getLogger(__name__).info(f"CSV file created at {output_csv_path} with {len(df)} entries.")
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save CSV file at {output_csv_path}: {e}")
        raise

    if azure_data:
        upload_csv_to_blob(str(output_csv_path), str(data_source))
        logging.getLogger(__name__).info(f"CSV file with the local path {output_csv_path} was saved on Azure filesystem {data_source} with {len(df)} entries.")
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create PH2 metadata CSV for SkiNet.")
    parser.add_argument("--azure-data",
                        action='store_true',
                        required=False,
                        help="If True, data_source is an AzureMachineLearningFileSystem instance, otherwise it is a local Path.")
    parser.add_argument("--data-source",
                        type=str,
                        required=True,
                        help="If using --azure-data, this must be the dataset name used in code (key from Azure YAML config). "
                        "If not, this must be the local path to the PH2 dataset root folder. Both given as string.")
    parser.add_argument("--output-csv",
                        type=str,
                        required=True,
                        help="Local path where the CSV file will be saved."
                        " If azure_data is True, this is the temporary local path where the CSV file will be saved before uploading to Azure.")

    args = parser.parse_args()
    if args.azure_data:
        data_source = args.data_source  # dataset name (str)
    else:
        data_source = Path(args.data_source)  # local path
    create_ph2_metadata(data_source, args.output_csv, azure_data=args.azure_data)
