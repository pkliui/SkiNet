from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Optional, Set

import pandas as pd
from pydantic import BaseModel, PrivateAttr

from SkiNet.Azure.azure_setup import AzureSetup
from SkiNet.ML.configs.datasets.dataset_keys import AzureDatasetKey


class BaseDataConfig(BaseModel):
    """
    Base class for dataset configuration.

    Args:
        csv_path (Optional[str]): Path to the CSV file. Can be absolute or relative to the current working directory (typically the repository root).
        azure_data (bool): Indicates if the data is stored in Azure Blob Storage.
            If True, the class will load the CSV from Azure using the Azure dataset key and CSV name supplied in the class variables of the subclass.
            If False, it will load from the local file system using csv_path that must be provided by user.

    Attributes:
        _data_frame (Optional[pd.DataFrame]): Cached DataFrame loaded from the CSV file.
            This is a private attribute and not part of model validation/serialization.
        REQUIRED_COLUMNS (Set[str]): Set of required columns for the dataset. Must be defined in subclasses.
        AZURE_DATASET_KEY (Optional[AzureDatasetKey]): One of the keys from AzureDatasetKey. Must match the key used in the YAML config file
        AZURE_CSV_NAME (Optional[str]): Name of the CSV file in Azure Blob Storage.

    Example usage (local CSV):
        cfg = MyDatasetConfig(csv_path="PH2Data/ph2_metadata.csv", azure_data=False)
        df = cfg.data_frame

    Example usage (Azure CSV):
        cfg = MyDatasetConfig(azure_data=True)
        df = cfg.data_frame

    Note:
        - For Azure, the value of the dataset key (AZURE_DATASET_KEY.value) must match the key in the YAML config file under PATH_ON_DATASTORE.
        - For local, csv_path must point to the actual CSV file (not a directory).
    """
    csv_path: Optional[str] = None
    azure_data: bool = False

    _data_frame: Optional[pd.DataFrame] = PrivateAttr(default=None)
    REQUIRED_COLUMNS: ClassVar[Set[str]] = set()
    AZURE_DATASET_KEY: ClassVar[Optional[AzureDatasetKey]] = None
    AZURE_CSV_NAME: ClassVar[Optional[str]] = None

    def validate_dataframe(self, df: pd.DataFrame, csv_path: str) -> None:
        """
        Validates the structure of the DataFrame by checking for required columns and empty DataFrames.

        :param df: The DataFrame to validate.
        :param csv_path: The path to the CSV file (for error messages).
        """
        # Check if DataFrame is empty
        if df.empty:
            raise ValueError(f"CSV at '{csv_path}' is empty.")
        # Check for missing columns
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"CSV at '{csv_path}' is missing required columns: {missing}")
        # Check if all values in required columns are empty or NaN
        if df[list(self.REQUIRED_COLUMNS)].replace("", pd.NA).isna().all().all():
            raise ValueError(f"CSV at '{csv_path}' is empty (all required columns have only empty values).")

    def read_csv(self, **kwargs: Any) -> pd.DataFrame:
        """Reads the CSV file (local or Azure) into a pandas DataFrame, validates, and stores it."""
        if self.azure_data:
            # get Azure file system
            AzureSetup.service_principal_authentication()
            dataset_name = self.AZURE_DATASET_KEY.value if self.AZURE_DATASET_KEY is not None else None
            logging.getLogger(__name__).info(f"Dataset_name on Azure as read from config is: {dataset_name}")
            assert isinstance(dataset_name, str)
            fs = AzureSetup.get_azureml_filesystem(dataset_name)

            # get the path to the dataset folder and CSV file on Azure
            _, path_on_azure = AzureSetup.get_azure_uri(dataset_name)

            if self.AZURE_CSV_NAME is None:
                raise ValueError("AZURE_CSV_NAME must be set for Azure data.")
            csv_path_on_azure = str(Path(path_on_azure) / self.AZURE_CSV_NAME)
            try:
                with fs.open(csv_path_on_azure, "r") as f:
                    df = pd.read_csv(f, **kwargs)
                self.validate_dataframe(df, csv_path_on_azure)
            except Exception as e:
                # AzureML may raise UserErrorException for missing files
                msg = f"CSV file not found on Azure at '{csv_path_on_azure}'."
                logging.getLogger(__name__).error(f"{msg} Original error: {e}")
                raise FileNotFoundError(f"{msg} Original error: {e}") from e
        else:
            if self.csv_path is None:
                raise ValueError("CSV path is not set.")
            df = pd.read_csv(self.csv_path, **kwargs)
            self.validate_dataframe(df, self.csv_path)
        self._data_frame = df
        return df

    @property
    def data_frame(self) -> pd.DataFrame:
        """Returns the loaded DataFrame, reading it if necessary."""
        if self._data_frame is None:
            self.read_csv()
        return self._data_frame
