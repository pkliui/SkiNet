from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Optional, Set

import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr

from SkiNet.Azure.azure_setup import AzureSetup
from SkiNet.ML.configs.datasets.dataset_keys import DatasetKey


class BaseDataConfig(BaseModel):
    """
    Base class for a dataset configuration.

    :attributes:
        METADATA_CSV_NAME (Optional[str]): Name of a dataset metadata file used in config, as defined in project paths. Must be specified in subclasses.
        REQUIRED_COLUMNS (Set[str]): Set of required columns in the metadata file. Must be specified in subclasses.
        AZURE_DATASET_KEY (Optional[DatasetKey]): One of the keys from DatasetKey. Must match the key used in the YAML config file and specified in subclasses.
        _metadata (Optional[pd.DataFrame]): Cached dataset metadata loaded from a CSV file into a DataFrame. Not part of model validation/serialization.

    Example usage (local CSV):
        cfg = MyDatasetConfig(local_data_root="some/local/path/to/data", azure_data=False)

    Example usage (Azure CSV):
        cfg = MyDatasetConfig(azure_data=True)
        # metadata dataframe is available through
        df = cfg.metadata

    Note:
        - For Azure, the value of the dataset key (AZURE_DATASET_KEY.value) must match the key in the YAML config file under PATH_ON_DATASTORE.
        - The CSV file must be present in the specified location (local or Azure) and must contain the required columns as per REQUIRED_COLUMNS.
    """
    azure_data: bool = Field(False, description="Indicates if the data is stored in Azure Blob Storage."
                             "If True, will load metadata CSV from Azure using the Azure dataset key and CSV name supplied in class variables of the subclass."
                             "If False, will load from the local file system using local_data_root that must be provided by user.")
    local_data_root: Optional[str] = Field(None, description="The root path to data and metadata locally. The path should point to a directory that contains"
                                           " folders with samples of data uniquely identifiable by their ID. Only used when no azure_data argument is set.")

    METADATA_CSV_NAME: ClassVar[Optional[str]] = None
    REQUIRED_COLUMNS: ClassVar[Set[str]] = set()
    AZURE_DATASET_KEY: ClassVar[Optional[DatasetKey]] = None

    _metadata: Optional[pd.DataFrame] = PrivateAttr(default=None)

    def _validate_config(self) -> None:
        """
        Validates the configuration by checking for required fields. Raises ValueError if required fields are missing based on the value of azure_data.
        """
        missing = []
        if self.azure_data:
            if self.AZURE_DATASET_KEY is None:
                missing.append("AZURE_DATASET_KEY")
            if self.METADATA_CSV_NAME is None:
                missing.append("METADATA_CSV_NAME")
            if missing:
                raise ValueError(f"Missing required Azure config values: {', '.join(missing)}")
        else:
            if self.local_data_root is None:
                missing.append("local_data_root")
            if self.METADATA_CSV_NAME is None:
                missing.append("METADATA_CSV_NAME")
            if missing:
                raise ValueError(f"Missing required local config values: {', '.join(missing)}")

    def _validate_dataframe(self, df: pd.DataFrame, csv_path: str) -> None:
        """
        Validates the structure of the read metadata DataFrame by checking for required columns and empty DataFrames.

        :param df: DataFrame to validate.
        :param csv_path: Path to the CSV file (for error messages).
        """
        if df.empty:
            raise ValueError(f"Metadata at '{csv_path}' is empty.")
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Metadata at '{csv_path}' is missing required columns: {missing}")
        # Check if all values in required columns are empty or NaN
        if df[list(self.REQUIRED_COLUMNS)].replace("", pd.NA).isna().all().all():
            raise ValueError(f"Metadata at '{csv_path}' is empty (all required columns have only empty values).")

    def read_metadata_csv(self, **kwargs: Any) -> pd.DataFrame:
        """
        Reads dataset metadata file (local or Azure) into a pandas DataFrame, validates, and returns it
        """
        self._validate_config()
        if self.azure_data:
            if self.local_data_root is not None:
                raise ValueError("Do not provide local_data_root argument pointing to a local dataset root when azure_data is True.")

            # get Azure file system
            AzureSetup.service_principal_authentication()
            dataset_key = self.AZURE_DATASET_KEY.value if self.AZURE_DATASET_KEY is not None else "local"
            logging.getLogger(__name__).info(f"Reading the following dataset on Azure: {dataset_key}")
            fs = AzureSetup.get_azureml_filesystem(dataset_key)
            _, data_root_on_azure = AzureSetup.get_azure_uri(dataset_key)

            # open metadata CSV file on Azure and read into DataFrame
            assert self.METADATA_CSV_NAME is not None
            csv_path_on_azure = str(Path(data_root_on_azure) / self.METADATA_CSV_NAME)
            try:
                with fs.open(csv_path_on_azure, "r") as f:
                    df = pd.read_csv(f, **kwargs)
                self._validate_dataframe(df, csv_path_on_azure)
            except Exception as e:
                # AzureML may raise UserErrorException for missing files
                msg = f"CSV file not found on Azure at '{csv_path_on_azure}'."
                logging.getLogger(__name__).error(f"{msg} Original error: {e}")
                raise FileNotFoundError(f"{msg} Original error: {e}") from e
        else:
            if self.local_data_root is None:
                raise ValueError("Provide local_data_root argument pointing to a local dataset root")

            # open metadata CSV file on local and read into DataFrame
            assert self.METADATA_CSV_NAME is not None
            csv_path_on_local = str(Path(self.local_data_root) / self.METADATA_CSV_NAME)
            df = pd.read_csv(csv_path_on_local, **kwargs)
            self._validate_dataframe(df, csv_path_on_local)
        self._metadata = df
        return df

    @property
    def metadata(self) -> pd.DataFrame:
        """Returns the loaded DataFrame, reading it if necessary."""
        if self._metadata is None:
            self.read_metadata_csv()
        return self._metadata
