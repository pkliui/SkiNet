from enum import Enum, unique


@unique
class DatasetKey(Enum):
    """
    Enum for dataset keys used to identify datasets in Azure.

    The VALUE of each enum member is a unique string identifier for data handled in this code and we call it `dataset_key`.
    The `dataset_key` must match one of the keys in PATH_ON_DATASTORE of your Azure YAML config.
    It is used to retrieve the correct path on the Azure datastore for that dataset.
    """
    PH2 = "PH2_DATASET"
