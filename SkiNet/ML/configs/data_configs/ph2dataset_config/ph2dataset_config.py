from typing import ClassVar, Literal, Set

from SkiNet.ML.configs.data_configs.base_data_config import BaseDataConfig
from SkiNet.ML.configs.datasets.dataset_keys import DatasetKey
from SkiNet.Utils.csv_headers import DATAPATH_HEADER, DATATYPE_HEADER, SAMPLEID_HEADER
from SkiNet.Utils.project_paths import PH2_CSV_NAME


class PH2DatasetConfig(BaseDataConfig):
    """
    Configuration for the PH2 dataset.

    Example usage for a local dataset:
    ```
    cfg = PH2DatasetConfig(csv_path="/PH2Data/ph2_metadata.csv", azure_data=False)
    df = cfg.data_frame
    ```

    Example usage for an Azure dataset:
    ```
    cfg = PH2DatasetConfig(azure_data=True)
    df = cfg.data_frame
    ```

    """
    kind: Literal["ph2"] = "ph2"
    REQUIRED_COLUMNS: ClassVar[Set[str]] = {SAMPLEID_HEADER, DATAPATH_HEADER, DATATYPE_HEADER}
    AZURE_DATASET_KEY: ClassVar[DatasetKey] = DatasetKey.PH2
    AZURE_CSV_NAME: ClassVar[str] = PH2_CSV_NAME
