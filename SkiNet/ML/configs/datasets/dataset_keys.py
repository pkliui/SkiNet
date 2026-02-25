from enum import Enum, unique


@unique
class DatasetKey(Enum):
    """
    Dataset keys. They uniquely identify keys in yaml files that map datasets stored in Azure.

    Example:
    ```
    from SkiNet.Azure.azure_setup import AzureSetup

    dataset_name = DatasetKey.PH2.value
    AzureSetup.service_principal_authentication()
    fs = AzureSetup.get_azureml_filesystem(dataset_name)
    ```
    """
    PH2 = "PH2_DATASET"
