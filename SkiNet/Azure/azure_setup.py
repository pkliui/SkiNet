import logging
import os
from pathlib import Path
from typing import Dict, Optional

from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azureml.fsspec import AzureMachineLearningFileSystem
from pydantic import BaseModel, Field, model_validator

from SkiNet.Utils import project_paths
from SkiNet.Utils.get_configs import get_config_from_yaml


class AzureSecrets(BaseModel):
    """
    Pydantic model for Azure secrets.
    """
    AZURE_CLIENT_SECRET: Optional[str] = Field(None, description="The password/credential generated for the service principal")

    @model_validator(mode="after")
    def check_secret(self) -> "AzureSecrets":
        if self.AZURE_CLIENT_SECRET is None or self.AZURE_CLIENT_SECRET.strip() == "":
            raise ValueError("AZURE_CLIENT_SECRET is missing or empty in the private Azure secrets YAML.")
        return self

    @classmethod
    def from_yaml(cls, path_to_yaml: Path) -> "AzureSecrets":
        """
        Load Azure secrets from a YAML file, e.g. the client secret for service principal authentication.

        :param path_to_yaml: Path to a YAML file with Azure secrets data.
            The YAML file should contain the following keys: AZURE_CLIENT_SECRET
        :return: Configured instance of AzureSecrets.
        """
        return cls(**get_config_from_yaml(path_to_yaml))

class AzureSetup(BaseModel):
    """
    Pydantic model for setting up Azure ML workspace and datastores from a YAML config.

    The YAML config file (e.g., azure_settings.yaml) must contain:
    - Azure credentials and workspace details with the following keys: AZURE_TENANT_ID, AZURE_CLIENT_ID,
    SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME, DATASTORE_NAME
    - A PATH_ON_DATASTORE dictionary mapping dataset enum KEY VALUES to their relative paths on the datastore

    Example YAML:
        AZURE_TENANT_ID: "<tenant-id>"
        AZURE_CLIENT_ID: "<client-id>"
        SUBSCRIPTION_ID: "<subscription-id>"
        RESOURCE_GROUP: "<resource-group>"
        WORKSPACE_NAME: "<workspace-name>"
        DATASTORE_NAME: "<datastore-name>"
        PATH_ON_DATASTORE:
          PH2_DATASET: "PH2DATA/"
          ANOTHER_DATASET: "another/path/"

    The yaml key PH2_DATASET must match the enum value of DatasetKey.PH2 if the dataset class is defined as

    @unique
    class DatasetKey(Enum):
        PH2 = "PH2_DATASET"
        ANOTHER = "ANOTHER_DATASET"

    Examples
    - Class initialisation from YAML:
    ```
    AzureSetup.from_yaml(path)
    ```
    - Building Azure ML datastore URIs for datasets
    ```
    azure_uri = AzureSetup.get_azure_uri("PH2_DATASET")
    ```
    - Accessing Azure ML datastores via fsspec
    ```
    fs = AzureSetup.get_azureml_filesystem("PH2_DATASET")
    ```
    """
    AZURE_TENANT_ID: Optional[str] = Field(None, description="The tenant ID returned when you created the service principal")
    AZURE_CLIENT_ID: Optional[str] = Field(None, description="The client ID returned when you created the service principal")
    SUBSCRIPTION_ID: Optional[str] = Field(None, description="Azure subscription ID")
    RESOURCE_GROUP: Optional[str] = Field(None, description="Resource group that contains the current workspace")
    WORKSPACE_NAME: Optional[str] = Field(None, description="Name of the workspace")
    DATASTORE_NAME: Optional[str] = Field(None, description="Name of the datastore containing data")
    PATH_ON_DATASTORE: Dict[str, str] = Field(default_factory=dict, description="Mapping of dataset keys to datastore paths")

    @model_validator(mode="after")
    def check_required_fields(self) -> "AzureSetup":
        def is_missing(value: str | None) -> bool:
            return value is None or (isinstance(value, str) and value.strip() == "")
        required = [
            "AZURE_TENANT_ID",
            "AZURE_CLIENT_ID",
            "SUBSCRIPTION_ID",
            "RESOURCE_GROUP",
            "WORKSPACE_NAME",
            "DATASTORE_NAME",
        ]
        missing = [field for field in required if is_missing(getattr(self, field))]
        if missing:
            raise ValueError(f"Missing required Azure config fields: {', '.join(missing)}")
        if is_missing(getattr(self, "PATH_ON_DATASTORE")):
            raise ValueError("PATH_ON_DATASTORE is missing or empty in the Azure config.")
        return self

    @classmethod
    def from_yaml(cls, path_to_yaml: Path) -> "AzureSetup":
        """
        Load Azure configuration from a YAML file.

        :param path_to_yaml: Path to a YAML file with Azure configuration data.
            The YAML file should contain the following keys: AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET,
            SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME, DATASTORE_NAME, PATH_ON_DATASTORE
        :return: Configured instance of AzureSetup.
        """
        return cls(**get_config_from_yaml(path_to_yaml))

    @classmethod
    def get_azure_uri(cls, dataset_name: str) -> str:
        """
        Build an Azure ML datastore URI for a configured dataset.

        :param dataset_name: The string value from DatasetKey enum (e.g., "PH2_DATASET").
            This must match one of the keys under PATH_ON_DATASTORE in your YAML config.

        :return: Azure ML URI string for the specified dataset

        Example return value:
        "azureml://subscriptions/{SUBSCRIPTION_ID}/resourcegroups/{RESOURCE_GROUP}/workspaces/{WORKSPACE_NAME}/datastores/{DATASTORE_NAME}/paths/{path}/",
        where, `{path}` is the value from PATH_ON_DATASTORE[dataset_name] in your YAML config.
        """
        azure_config = cls.from_yaml(project_paths.AZURE_SETTINGS_YAML)
        data_root_on_azure = cls.get_rel_data_root_on_azure(dataset_name)

        azure_uri = f"azureml://subscriptions/{azure_config.SUBSCRIPTION_ID}/resourcegroups/{azure_config.RESOURCE_GROUP}/workspaces/{azure_config.WORKSPACE_NAME}/datastores/{azure_config.DATASTORE_NAME}/paths/{data_root_on_azure}/"  # noqa: E501
        return azure_uri

    @classmethod
    def get_rel_data_root_on_azure(cls, dataset_name: str) -> str:
        """
        Get relative path to a dataset root directory on the datastore.

        :param dataset_name: The string value from DatasetKey enum (e.g., "PH2_DATASET").
            This must match one of the keys under PATH_ON_DATASTORE in your YAML config.

        :return: The relative path to a dataset root directory on the datastore.

        Example return value: "{data_root_on_azure}"
        Here, `{data_root_on_azure}` is the value from PATH_ON_DATASTORE[dataset_name] in your YAML config.
        """
        azure_config = cls.from_yaml(project_paths.AZURE_SETTINGS_YAML)

        # get path to data on the datastore w.r.t. its root
        if dataset_name not in azure_config.PATH_ON_DATASTORE:
            raise ValueError(f"The keys of azure_config.PATH_ON_DATASTORE do not contain '{dataset_name}'. "
                             f"Check keys in YAML config {project_paths.AZURE_SETTINGS_YAML} and make sure they match the DatasetKey enum values."
                             f"Available options: {list(azure_config.PATH_ON_DATASTORE.keys())}")
        data_root_on_azure = azure_config.PATH_ON_DATASTORE[dataset_name]
        return data_root_on_azure

    @classmethod
    def get_azureml_filesystem(cls, dataset_name: str) -> AzureMachineLearningFileSystem:
        """
        Get AzureMachineLearningFileSystem for a dataset.

        :param dataset_name: String key from PATH_ON_DATASTORE in your Azure YAML config (see get_azure_uri docstring for example).
        :return: AzureMachineLearningFileSystem object for the specified dataset.
        """
        azure_uri = cls.get_azure_uri(dataset_name)
        fs = AzureMachineLearningFileSystem(azure_uri)
        return fs


def service_principal_authentication() -> DefaultAzureCredential:
    """
    Perform service principal (SP) authentication.
    Used for local development and other environments where you can securely store SP credentials.

    Service Principal (SP) uses the Azure Identity package for Python.
    DefaultAzureCredential class looks for the following environment variables and uses their values
    when authenticating as the SP: AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_SECRET. See more at:
    1. https://learn.microsoft.com/en-us/entra/identity-platform/app-objects-and-service-principals?tabs=browser
    2. https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication?view=azureml-api-2&tabs=sdk
    3. https://devblogs.microsoft.com/azure-sdk/authentication-and-the-azure-sdk/
    """

    azure_config = AzureSetup.from_yaml(project_paths.AZURE_SETTINGS_YAML)
    logging.getLogger(__name__).info(f"Loaded Azure configuration from {project_paths.AZURE_SETTINGS_YAML}")

    azure_secrets = AzureSecrets(**get_config_from_yaml(project_paths.PRIVATE_AZURE_SECRETS_YAML))
    logging.getLogger(__name__).info(f"Loaded Azure secrets from {project_paths.PRIVATE_AZURE_SECRETS_YAML}")

    # set environment variables needed to perform Azure authentication using DefaultAzureCredential
    logging.getLogger(__name__).info("Setting environment variables AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET")
    os.environ["AZURE_TENANT_ID"] = azure_config.AZURE_TENANT_ID or ""
    os.environ["AZURE_CLIENT_ID"] = azure_config.AZURE_CLIENT_ID or ""
    os.environ["AZURE_CLIENT_SECRET"] = azure_secrets.AZURE_CLIENT_SECRET or ""

    # The DefaultAzureCredential tries to infer what environment is being used and uses the most appropriate credential.
    # Env variables set above are its 1st choice and will be used along with Azure Active Directory to authenticate the connection.
    # These environment variables define the service principal that was granted role-based permission to access data.
    #
    # NB: SP authentication is available for local development! Alternatively, one may want to use Managed Identity,
    # which would be the next choice for DefaultAzureCredential. Using Managed Identities eliminates the need to manage credentials,
    # but this option is apparently available only for Azure machines!

    return DefaultAzureCredential()

def managed_identity_authentication() -> ManagedIdentityCredential:
    """
    Authenticate using Azure Managed Identity.
    Used to access Azure resources like Azure Blob Storage without the need for explicit credentials.
    For example, in Azure Compute instances.
    """
    client_id = (
        os.getenv("AZURE_MANAGED_IDENTITY_CLIENT_ID", "").strip()
        or os.getenv("DEFAULT_IDENTITY_CLIENT_ID", "").strip()
    )

    if client_id:
        logging.getLogger(__name__).info("Using user-assigned managed identity")
        return ManagedIdentityCredential(client_id=client_id)

    logging.getLogger(__name__).info("Using system-assigned managed identity")
    return ManagedIdentityCredential()
