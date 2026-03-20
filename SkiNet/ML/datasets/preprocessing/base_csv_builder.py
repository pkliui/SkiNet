import argparse
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
from azureml.fsspec import AzureMachineLearningFileSystem
from numpy.typing import NDArray

from SkiNet.Azure.azure_setup import AzureSetup, service_principal_authentication
from SkiNet.ML.utils.data_utils import convert_to_numpy_bytes, filter_missing_images_and_masks
from SkiNet.Utils.csv_headers import DATAPATH_HEADER, DATATYPE_HEADER, DATATYPE_IMAGE, DATATYPE_MASK, SAMPLEID_HEADER


class BaseCSVBuilder(ABC):
    """
    Generic abstract base class for building CSV metadata for datasets, supporting both local and Azure data sources.

    Expected to be called on local due to service principal authentication in AzureMachineLearningFileSystem,
    but the design allows for flexibility in supporting both environments if refactored.
    """

    @property
    @abstractmethod
    def data_root(self) -> Union[Path, AzureMachineLearningFileSystem]:
        """
        Return the root object for data access (Path for local, fs for Azure).
        Must be implemented in environment-specific subclasses (e.g., LocalCSVBuilder, AzureCSVBuilder).
        """
        pass

    @property
    @abstractmethod
    def image_pattern(self) -> str:
        """Glob pattern to match image files. Must be defined in dataset-specific subclasses (e.g., PH2BaseCSVBuilder)"""
        pass

    @property
    @abstractmethod
    def mask_pattern(self) -> str:
        """Glob pattern to match mask files. Must be defined in dataset-specific subclasses."""
        pass

    @property
    @abstractmethod
    def output_csv_name(self) -> str:
        """Name of the output CSV file. Must be defined in dataset-specific subclasses."""
        pass

    @abstractmethod
    def sampleid_func(self, path_str: str) -> str:
        """
        Function to extract sample ID from a path string. Must be implemented in dataset-specific subclasses,
        as the logic for extracting a sample ID depends on the dataset's file structure.
        """
        pass

    @abstractmethod
    def datapath_func(self, path_str: str) -> str:
        """
        Function to extract a path relative to the dataset root from a full path.
        Must be implemented in environment-specific subclasses (e.g., LocalCSVBuilder, AzureCSVBuilder).
        """
        pass

    @abstractmethod
    def create_metadata_csv(self) -> None:
        """
        Create a metadata CSV file.
        Concrete implementation must be provided in dataset-specific subclasses (e.g., PH2BaseCSVBuilder)
        """
        pass

    def get_data_paths(self) -> Tuple[NDArray[np.bytes_], NDArray[np.bytes_]]:
        """
        Get paths to images and masks for a dataset in the given data root and with the specified patterns.

        :return: A tuple of two numpy arrays containing the paths to images and masks, respectively, converted to bytes.
        """
        return self.get_image_and_mask_paths(self.data_root, self.image_pattern, self.mask_pattern)

    def get_image_and_mask_paths(self,
                                 data_root: Union[Path, AzureMachineLearningFileSystem],
                                 image_pattern: str,
                                 mask_pattern: str) -> Tuple[NDArray[np.bytes_], NDArray[np.bytes_]]:
        """
        Get paths to images and masks of a dataset given a data root and glob patterns for images and masks.
        The method supports both local file systems (using Path) and Azure file systems (using AzureMachineLearningFileSystem).

        :param data_root: if provided as a Path, it is a local directory that contains folders with samples of data uniquely identifiable by their ID,
            otherwise it is an Azure Machine Learning filesystem instance referencing a specific location on Azure data storage.
        :param image_pattern: A glob pattern to match image files in the dataset.
        :param mask_pattern: A glob pattern to match mask files in the dataset.
        :return: A tuple of two numpy arrays of dtype np.bytes_ that contains full paths to images and masks.
            This data type is required by the Dataset class to fix the copy-on-write problem that results in an increased memory usage
            https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
        """
        image_paths = list(data_root.glob(image_pattern))
        mask_paths = list(data_root.glob(mask_pattern))

        image_paths, mask_paths = filter_missing_images_and_masks(image_paths, mask_paths)

        # Convert lists to NumPy arrays of dtype np.bytes_ to fix copy-on-write problem that results in an increased memory usage
        # https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
        return convert_to_numpy_bytes(image_paths, mask_paths)

    def create_dataframe_with_paths_and_types(self,
                                              image_paths: NDArray[np.bytes_],
                                              mask_paths: NDArray[np.bytes_],
                                              sampleid_func: Callable[[str], str],
                                              datapath_func: Callable[[str], str]) -> pd.DataFrame:
        """
        Construct a metadata DataFrame from arrays of image and mask paths.
        Each row contains the sample ID, relative data path, and data type ("image" or "mask").
        The sample ID and relative path are extracted using the provided functions.

        :param image_paths: NumPy array of dtype np.bytes_ containing full paths to images.
        :param mask_paths: NumPy array of dtype np.bytes_ containing full paths to masks.
        :param sampleid_func: Function that takes a file path (absolute or relative, as a string) as input
            and returns the corresponding sample ID as a string.
        :param datapath_func: Function that takes a path as input and returns the path relative to the dataset root.
        :return: DataFrame containing basic metadata for the dataset.
        """
        data = []
        for img_path, mask_path in zip(image_paths, mask_paths):
            # Decode bytes to string if necessary
            img_path_str = img_path.decode("utf-8") if isinstance(img_path, bytes) else str(img_path)
            mask_path_str = mask_path.decode("utf-8") if isinstance(mask_path, bytes) else str(mask_path)

            # Extract sample ID and relative paths using the provided functions
            sample_id = sampleid_func(img_path_str)
            img_rel_path = datapath_func(img_path_str)
            mask_rel_path = datapath_func(mask_path_str)

            # Add entries for both image and mask to the data list
            data.append({SAMPLEID_HEADER: sample_id, DATAPATH_HEADER: img_rel_path, DATATYPE_HEADER: DATATYPE_IMAGE})
            data.append({SAMPLEID_HEADER: sample_id, DATAPATH_HEADER: mask_rel_path, DATATYPE_HEADER: DATATYPE_MASK})
        return pd.DataFrame(data)

    def create_basic_metadata(self) -> pd.DataFrame:
        """
        Generate a basic metadata DataFrame for the dataset. This method retrieves image and mask paths using the dataset's configuration,
        then builds a DataFrame with sample IDs, relative paths, and data types by calling `create_dataframe_with_paths_and_types`.

        Expected to be called in dataset-specific subclasses that implement the concrete logic
        for accessing the dataset (e.g. PH2BaseCSVBuilder)

        :return: DataFrame containing the basic metadata structure
        """
        image_paths, mask_paths = self.get_data_paths()
        return self.create_dataframe_with_paths_and_types(image_paths, mask_paths, self.sampleid_func, self.datapath_func)

    def save_dataframe_to_csv(self, df: pd.DataFrame, output_csv_path: Union[str, Path]) -> None:
        """
        Save a DataFrame to a CSV file locally.

        :param df: DataFrame to save.
        :param output_csv_path: Local path where the CSV file will be saved.
        """
        try:
            df.to_csv(output_csv_path, index=False)
            logging.getLogger(__name__).info(f"CSV file created at {output_csv_path} with {len(df)} entries.")
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to save CSV file at {output_csv_path}: {e}")
            raise

class LocalCSVBuilder(BaseCSVBuilder):
    """
    Base class for building local CSV metadata.
    """

    def __init__(self, arg: argparse.Namespace):
        """
        :param arg: Command-line arguments containing the root path to data on a local file system.
        """
        if Path(arg.local_data_root).is_dir():
            self._full_path_to_local_data_root = Path(arg.local_data_root)
            """Path to the local data root directory containing folders with samples of data uniquely identifiable by their ID."""
        else:
            logging.getLogger(__name__).error(f"The provided local data root path {arg.local_data_root} is not a directory.")
            raise ValueError(f"The provided local data root path {arg.local_data_root} is not a directory.")

    @property
    def data_root(self) -> Path:
        """
        Return the root directory of the local dataset as a Path object, based on the command-line arguments.

        Note: Always use the data_root property to access the local dataset root, rather than directly accessing the internal attribute.
        """
        return self._full_path_to_local_data_root

    def datapath_func(self, path_str: str) -> str:
        """
        Given a local file path, return path relative to the dataset root.

        :param path_str: Local path to an image or a mask file.
        :return: Path relative to the dataset root as a string

        Example:
            path_str = "/local_path/dataset_root_folder/sample_folder/image.jpg"
            Given data_root="/local_path/dataset_root_folder", this function will return "sample_folder/image.jpg".
        """
        return str(Path(path_str).relative_to(self.data_root))


class AzureCSVBuilder(BaseCSVBuilder):
    """
    Base class for building Azure CSV metadata.

    Expected to be called on local due to service principal authentication in AzureMachineLearningFileSystem,
    but the design allows for flexibility in supporting both environments if refactored.
    """

    def __init__(self, dataset_name: str):
        """
        :param dataset_name: One of the dataset names from DatasetKey enum or a YAML file that maps to a dataset path on Azure.
            The value of the dataset name must match the key in the YAML config file under PATH_ON_DATASTORE.
        """
        service_principal_authentication()
        self.fs = AzureSetup.get_azureml_filesystem(dataset_name)
        self._data_root_on_azure = AzureSetup.get_rel_data_root_on_azure(dataset_name)

    @property
    def data_root_on_azure(self) -> str:
        """
        Returns the root path to the dataset on Azure (as specified in the YAML config).
        """
        return self._data_root_on_azure

    @property
    def data_root(self) -> AzureMachineLearningFileSystem:
        """
        Return Azure Machine Learning filesystem instance. In contrast to the local data root, which is a directory path,
        the Azure data root is an AzureMachineLearningFileSystem instance that is used to interact with the Azure Blob Storage.

        Note 1: For specifying paths on Azure, use the `data_root_on_azure` property.

        Note 2: even if the filesystem is mounted to the dataset root (i.e. self.fs = AzureSetup.get_azureml_filesystem(dataset_name)),
        glob() and ls() may still return paths prefixed with the datastore root path.  For this reason, use the `datapath_func` method
        to obtain paths relative to the dataset root.
        """
        try:
            return self.fs
        except Exception as e:
            raise ValueError(f"Error occurred while accessing Azure file system: {e}")

    def datapath_func(self, path_str: str) -> str:
        """
        Given a file path, return path relative to the dataset root on Azure.

        :param path_str: Azure path to an image or a mask file,
            as obtained with e.g. AzureMachineLearningFileSystem.glob() or AzureMachineLearningFileSystem.ls().
        :return: Path relative to the dataset root as a string

        Example:
        path_str = "dataset_root_folder/sample_folder/image.jpg"
        Given data_root_on_azure="dataset_root_folder", this function will return "sample_folder/image.jpg".
        """
        root = str(self.data_root_on_azure).rstrip('/')
        if path_str.startswith(root + '/'):
            return path_str[len(root) + 1:]
        elif path_str == root:
            return ""
        return path_str

    def upload_csv_to_blob(self, local_csv_path: str) -> None:
        """
        Upload a local CSV file to Azure Blob Storage using AzureSetup authentication.

        :param local_csv_path: Local path to the CSV file to upload to Azure Blob Storage.
        """
        # recursive must be set to False to upload a file
        self.fs.upload(lpath=local_csv_path, rpath=self.data_root_on_azure, recursive=False, **{'overwrite': 'MERGE_WITH_OVERWRITE'})

    def save_dataframe_and_upload_csv(self, df: pd.DataFrame) -> None:
        """
        Save a DataFrame as a temporary CSV file locally and then upload it to Azure Blob Storage.

        :param df: The DataFrame to save in Azure.
        """
        tmp_dir = tempfile.gettempdir()
        temp_csv_path = os.path.join(tmp_dir, self.output_csv_name)
        self.save_dataframe_to_csv(df=df, output_csv_path=temp_csv_path)

        try:
            self.upload_csv_to_blob(local_csv_path=temp_csv_path)
            logging.getLogger(__name__).info(f"CSV file uploaded to Azure path {self.data_root_on_azure}")
        finally:
            try:
                os.remove(temp_csv_path)
                logging.getLogger(__name__).info(f"Temporary file {temp_csv_path} deleted.")
            except Exception as e:
                logging.getLogger(__name__).warning(f"Could not delete temporary file {temp_csv_path}: {e}")
