from typing import ClassVar, Literal, Set

from pydantic import Field

from SkiNet.ML.configs.data_configs.base_data_config import BaseDataConfig
from SkiNet.ML.configs.datasets.dataset_keys import DatasetKey
from SkiNet.Utils.csv_headers import DATAPATH_HEADER, DATATYPE_HEADER, SAMPLEID_HEADER
from SkiNet.Utils.project_paths import PH2_CSV_NAME


class PH2DatasetConfig(BaseDataConfig):
    """
    Configuration for the PH2 dataset.

    :attributes:
        REQUIRED_COLUMNS (ClassVar[Set[str]]): Set of required columns in the metadata CSV.
        AZURE_DATASET_KEY (ClassVar[DatasetKey]): Key for the Azure dataset.
        METADATA_CSV_NAME (ClassVar[str]): Name of the metadata CSV file as defined in project paths.

    Example usage for a local dataset:
    ```
    cfg = PH2DatasetConfig(local_data_root="local_path/PH2Data", azure_data=False)
    df = cfg.metadata
    ```

    Example usage for an Azure dataset:
    ```
    cfg = PH2DatasetConfig(azure_data=True)
    df = cfg.metadata
    ```

    """
    kind: Literal["ph2"] = Field("ph2", description="Dataset kind identifier for PH2. Used for config selection and validation.")

    REQUIRED_COLUMNS: ClassVar[Set[str]] = {SAMPLEID_HEADER, DATAPATH_HEADER, DATATYPE_HEADER}
    AZURE_DATASET_KEY: ClassVar[DatasetKey] = DatasetKey.PH2
    METADATA_CSV_NAME: ClassVar[str] = PH2_CSV_NAME
