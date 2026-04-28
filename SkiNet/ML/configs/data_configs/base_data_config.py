from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Optional, Set

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from SkiNet.Azure.azure_setup import AzureSetup
from SkiNet.Utils import project_paths
from SkiNet.Utils.experiment_keys import DatasetKey
from SkiNet.Utils.data.split_data import SplitConfig


class BaseDataConfig(BaseModel):
    """
    Base class to configure dataset from metadata

    :param azure_data: Whether data and metadata reside in Azure Blob Storage.
        If True, metadata is read directly from a mounted Azure Blob Storage, assuming
        the appropriate Managed Identity permissions and configurations are in place.
        If False, metadata and files are read from a local_data_root.
    :param azure_blob_mount_point: The mount point for the Azure Blob Storage (if using Azure).
        This is not a path to data and CSV file, but the root mount point for the blob storage.
        The actual path to the dataset and metadata CSV within the blob storage is determined by the DATASET_KEY and AZURE_SETTINGS_YAML config.
    :param local_data_root: The root path to the local data (if not using Azure).
        This is the directory where all data and the metadata CSV file resides.

    :attributes:
        METADATA_CSV_NAME (Optional[str]): Name of a dataset metadata file used in config, as defined in project paths. Must be specified in subclasses.
        REQUIRED_COLUMNS (Set[str]): Set of required columns in the metadata file. Must be specified in subclasses.
        DATASET_KEY (Optional[DatasetKey]): One of the keys from DatasetKey.
            Its value must match the key used in the YAML config file and specified in subclasses.
        _metadata (Optional[pd.DataFrame]): Cached dataset metadata loaded from a CSV file into a DataFrame. Not part of model validation/serialization.

    Example usage (local CSV):
        cfg = MyDatasetConfig(local_data_root="some/local/path/to/data", azure_data=False)

    Example usage (Azure CSV):
        cfg = MyDatasetConfig(azure_blob_mount_point="mnt/data", azure_data=True)

        # in both cases, metadata dataframe is available through
        df = cfg.metadata

    Note:
        - For Azure, the value of the dataset key (DATASET_KEY.value) must match the key in the YAML config file under PATH_ON_DATASTORE.
        - The CSV file must be present in the specified location (local or Azure) and must contain the required columns as per REQUIRED_COLUMNS.
    """
    model_config = ConfigDict(extra='ignore', validate_assignment=True)

    azure_data: bool = Field(False, description="Indicates if the data is stored in Azure Blob Storage."
                             "If True, requires azure_blob_mount_point to be provided by user."
                             "If False, local_data_root must be provided by user.")
    azure_blob_mount_point: Optional[str] = Field(
        None, description="The mount point for the Azure Blob Storage. Required if azure_data is True. Ignored if azure_data is False.")
    local_data_root: Optional[str] = Field(None, description="The root path to data and metadata locally. The path should point to a directory that contains"
                                           " folders with samples of data uniquely identifiable by their ID. Only used when no azure_data argument is set.")
    split_train_size: float = Field(default=0.6, ge=0.0, le=1.0, description="Proportion of the dataset in the train split.")
    split_val_size: float = Field(default=0.2, ge=0.0, le=1.0, description="Proportion of the dataset in the validation split.")
    split_test_size: float = Field(default=0.2, ge=0.0, le=1.0, description="Proportion of the dataset in the test split.")
    split_random_seed: int = Field(default=42, ge=0, description="Random seed of the train/val/test splits.")
    split_stratify_column: str | None = Field(default=None, description="Column name in the metadata CSV to use for"
                                              "stratified splitting into train/val/test splits. Should be a column  "
                                              "in the metadata CSV that has categorical labels for stratification. ")

    METADATA_CSV_NAME: ClassVar[str]
    REQUIRED_COLUMNS: ClassVar[Set[str]] = set()
    DATASET_KEY: ClassVar[Optional[DatasetKey]] = None

    _metadata: Optional[pd.DataFrame] = PrivateAttr(default=None)

    def _validate_config(self) -> None:
        """
        Validates the configuration by checking for required fields.
        Raises ValueError if required fields are missing based on the value of azure_data.
        """
        missing = []

        if self.azure_data:
            if self.azure_blob_mount_point is None:
                missing.append("azure_blob_mount_point")
            if self.DATASET_KEY is None:
                missing.append("DATASET_KEY")
        else:
            if self.local_data_root is None:
                missing.append("local_data_root")

        if self.METADATA_CSV_NAME is None:
            missing.append("METADATA_CSV_NAME")
        if self.REQUIRED_COLUMNS is None:
            missing.append("REQUIRED_COLUMNS")
        if missing:
            raise ValueError(f"Missing required config values: {', '.join(missing)}")

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

        # Authentication and access to Azure Blob Storage are assumed to be handled externally,
        # e.g. through appropriate Managed Identity permissions and Azure setup.
        if self.azure_data and self.local_data_root is not None:
            logging.getLogger(__name__).warning(
                "Both azure_data is True and local_data_root is provided. local_data_root will be ignored.")
        if not self.azure_data and self.local_data_root is None:
            raise ValueError("Provide local_data_root argument pointing to a local dataset root")

        assert self.METADATA_CSV_NAME is not None
        assert self.data_root is not None

        # read CSV
        csv_path = str(Path(self.data_root) / self.METADATA_CSV_NAME)
        df = pd.read_csv(csv_path, **kwargs)

        self._validate_dataframe(df, csv_path)
        self._metadata = df
        return df

    @property
    def metadata(self) -> pd.DataFrame:
        """Returns the loaded DataFrame with metadata, reading it if necessary."""
        if self._metadata is None:
            self.read_metadata_csv()
        return self._metadata

    @property
    def data_root(self) -> Path:
        """
        Returns the root path for the dataset and metadata CSV, taking into account whether it is stored on Azure or locally.

        Note DATASET_KEY specified in child classes uniquely identify the dataset. This key is used to locate
        the dataset within the blob storage, as specified in AZURE_SETTINGS_YAML.

        :return: The root path for the dataset either local or mounted Azure Blob Storage,
        given its key specified in the relevant child class and AZURE_SETTINGS_YAML config.
        """
        self._validate_config()

        if self.azure_data:
            assert self.azure_blob_mount_point is not None
            assert self.DATASET_KEY is not None

            azure_config = AzureSetup.from_yaml(project_paths.AZURE_SETTINGS_YAML)
            relative_dataset_root = azure_config.PATH_ON_DATASTORE[self.DATASET_KEY.value]
            return Path(self.azure_blob_mount_point) / Path(relative_dataset_root)
        else:
            assert self.local_data_root is not None
            return Path(self.local_data_root)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> "BaseDataConfig":
        """
        Create a deep copy of the BaseDataConfig instance.

        This override ensures that the internal `_metadata` cache dataframe is not copied to the new
        instance. Instead, the copied object resets `_metadata` to `None`, so
        each consumer can lazily load its own metadata independently.

        :param memo: Dictionary used by the deepcopy mechanism to track
            already-copied objects and avoid infinite recursion.

        :return: A deep-copied instance with `_metadata` cleared.
        """
        copy = super().__deepcopy__(memo)
        # Don't carry the cached DataFrame into each copy; each consumer lazy-loads independently.
        copy._metadata = None
        return copy

    def get_split_config(self) -> SplitConfig:
        """
        Return a SplitConfig object containing the train/val/test split proportions,
        random seed, and stratification column as per the configuraion
        """
        return SplitConfig(train_size=self.split_train_size,
                           val_size=self.split_val_size,
                           test_size=self.split_test_size,
                           stratify_column=self.split_stratify_column,
                           random_seed=self.split_random_seed)
