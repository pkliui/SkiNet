from enum import Enum, unique


@unique
class AzureDatasetKey(Enum):
    """
    Azure dataset keys. They uniquely identify keys in yaml files that map  datasets stored in Azure.

    Example:
    ```
    from SkiNet.Azure.azure_setup import AzureSetup

    dataset_name = AzureDatasetKey.PH2_DATASET_KEY.value
    AzureSetup.service_principal_authentication()
    fs = AzureSetup.get_azureml_filesystem(dataset_name)
    ```
    """
    PH2_DATASET_KEY = "PH2_DATASET"
