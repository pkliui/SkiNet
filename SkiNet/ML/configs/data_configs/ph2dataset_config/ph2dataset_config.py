from typing import ClassVar, Literal, Set
from enum import Enum, unique

from pydantic import Field

from SkiNet.ML.configs.data_configs.base_data_config import BaseDataConfig
from SkiNet.Utils.csv_headers import DATAPATH_HEADER, DATATYPE_HEADER, SAMPLEID_HEADER, PH2_CLINICAL_DIAGNOSIS_HEADER, PH2_COLORS_HEADER
from SkiNet.Utils.experiment_keys import DatasetKey
from SkiNet.Utils.project_paths import PH2_CSV_NAME
from SkiNet.Utils.data.split_data import SplitConfig


@unique
class PH2StratificationOptions(str, Enum):
    """
    Enum for PH2 dataset stratification options.
    These are the possible values for the "stratify_by" argument in the split_segmentation_metadata function
    for the PH2 dataset, which is used to specify the column by which to stratify the data for train/val/test sets.
    """
    PH2_CLINICAL_DIAGNOSIS = PH2_CLINICAL_DIAGNOSIS_HEADER
    PH2_COLORS = PH2_COLORS_HEADER


class PH2DatasetConfig(BaseDataConfig):
    """
    Configuration for the PH2 dataset.

    :attributes:
        REQUIRED_COLUMNS (ClassVar[Set[str]]): Set of required columns in the metadata CSV.
        DATASET_KEY (ClassVar[DatasetKey]): Key for the Azure dataset.
        METADATA_CSV_NAME (ClassVar[str]): Name of the metadata CSV file as defined in project paths.

    Example usage for a local dataset:
    ```
    cfg = PH2DatasetConfig(local_data_root="local_path/PH2Data", azure_data=False)
    df = cfg.metadata
    ```

    Example usage for an Azure dataset:
    ```
    cfg = PH2DatasetConfig(azure_blob_mount_point="mnt/data", azure_data=True)
    df = cfg.metadata
    ```

    """
    kind: Literal["ph2"] = Field("ph2", description="Dataset kind identifier for config selection and validation.")
    split_stratify_column: PH2StratificationOptions | None = Field(
        default=PH2StratificationOptions.PH2_CLINICAL_DIAGNOSIS,
        description="Column name in the metadata CSV to use for stratified splitting into train/val/test splits. "
        "Should be a column that exists in the metadata CSV and contains categorical labels for stratification. "
        "Set to None to disable stratification. For PH2, we use PH2StratificationOptions for stratification."
    )

    REQUIRED_COLUMNS: ClassVar[Set[str]] = {SAMPLEID_HEADER, DATAPATH_HEADER, DATATYPE_HEADER}
    DATASET_KEY: ClassVar[DatasetKey] = DatasetKey.PH2
    METADATA_CSV_NAME: ClassVar[str] = PH2_CSV_NAME

    def get_split_config(self) -> SplitConfig:
        """
        Gets the SplitConfig for the PH2 dataset based on the specific split column and
        the train/val/test sizes and random seed defined in the BaseDataConfig.

        :returns: SplitConfig for the PH2 dataset, where the stratify_column is PH2-specific when enabled
        """
        return SplitConfig(train_size=self.split_train_size,
                           val_size=self.split_val_size,
                           test_size=self.split_test_size,
                           stratify_column=self.split_stratify_column.value if self.split_stratify_column is not None else None,
                           random_seed=self.split_random_seed)
