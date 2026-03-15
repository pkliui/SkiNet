import logging
import os
from pathlib import Path
from typing import Any, Union

import param
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azureml.fsspec import AzureMachineLearningFileSystem, ManagedIdentityCredential

from SkiNet.Utils import project_paths
from SkiNet.Utils.get_configs import get_config_from_yaml


class AzureSetup(param.Parameterized):
    """
    Parametrized class for Azure setup and authentication.

    Loads Azure configuration from YAML files and provides methods for:
    - Service principal authentication (using DefaultAzureCredential or InteractiveBrowserCredential)
    - Building Azure ML datastore URIs for datasets
    - Accessing Azure ML datastores via fsspec

    The YAML config file (e.g., azure_settings.yaml) should contain:
        - Azure credentials and workspace details
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
    """
    AZURE_TENANT_ID: str = param.String(doc="The tenant ID returned when you created the service principal")
    AZURE_CLIENT_ID: str = param.String(doc="The client ID returned when you created the service principal")
    AZURE_CLIENT_SECRET: str = param.String(doc="The password/credential generated for the service principal")
    SUBSCRIPTION_ID: str = param.String(doc="Azure subscription ID")
    RESOURCE_GROUP: str = param.String(doc="Resource group that contains the current workspace")
    WORKSPACE_NAME: str = param.String(doc="Name of the workspace")
    DATASTORE_NAME: str = param.String(doc="Name of the datastore containing data")
    PATH_ON_DATASTORE = param.Dict(doc="Dictionary mapping DatasetKey enum values to paths on datastore, as defined in azure_settings.yaml")

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

    @classmethod
    def get_azure_config_from_yaml(cls, path_to_yaml: Path) -> "AzureSetup":
        """
        Load Azure configuration from a YAML file.

        :param path_to_yaml: Path to a YAML file with Azure configuration.
        :return: Configured instance of AzureSetup.

        Expected YAML structure is described in AzureSetup's docstring
        """
        return cls(**get_config_from_yaml(path_to_yaml))

    @classmethod
    def get_azure_uri(cls, dataset_name: str) -> tuple[str, str]:
        """
        Build an Azure ML datastore URI for a configured dataset.

        :param dataset_name: The string value from DatasetKey enum (e.g., "PH2_DATASET").
            This must match one of the keys under PATH_ON_DATASTORE in your YAML config.

        :return: Tuple of (Azure ML URI string for the specified dataset, relative path to a dataset root directory on the datastore).

        Example return value:
            (
                "azureml://subscriptions/{SUBSCRIPTION_ID}/resourcegroups/{RESOURCE_GROUP}/workspaces/{WORKSPACE_NAME}/datastores/{DATASTORE_NAME}/paths/{path}/",
                "{data_root_on_azure}"
            )
        Here, `{data_root_on_azure}` is the value from PATH_ON_DATASTORE[dataset_name] in your YAML config.
        """
        azure_config = cls.get_azure_config_from_yaml(project_paths.AZURE_SETTINGS_YAML)

        # get path to data on the datastore w.r.t. its root
        if dataset_name not in azure_config.PATH_ON_DATASTORE:
            raise ValueError(f"The keys of azure_config.PATH_ON_DATASTORE do not contain '{dataset_name}'. "
                             f"Check keys in YAML config {project_paths.AZURE_SETTINGS_YAML} and make sure they match the DatasetKey enum values."
                             f"Available options: {list(azure_config.PATH_ON_DATASTORE.keys())}")
        data_root_on_azure = azure_config.PATH_ON_DATASTORE[dataset_name]

        azure_uri = f"azureml://subscriptions/{azure_config.SUBSCRIPTION_ID}/resourcegroups/{azure_config.RESOURCE_GROUP}/workspaces/{azure_config.WORKSPACE_NAME}/datastores/{azure_config.DATASTORE_NAME}/paths/{data_root_on_azure}/"  # noqa: E501
        return (azure_uri, data_root_on_azure)

    @classmethod
    def get_azureml_filesystem(cls, dataset_name: str) -> AzureMachineLearningFileSystem:
        """
        Get AzureMachineLearningFileSystem for a dataset.

        :param dataset_name: String key from PATH_ON_DATASTORE in your Azure YAML config (see get_azure_uri docstring for example).
        :return: AzureMachineLearningFileSystem object for the specified dataset.
        """
        azure_uri, _ = cls.get_azure_uri(dataset_name)
        fs = AzureMachineLearningFileSystem(azure_uri)
        return fs

    @classmethod
    def service_principal_authentication(cls) -> Union[DefaultAzureCredential, InteractiveBrowserCredential, ManagedIdentityCredential]:
        """
        Return an Azure credential usable for both local and Azure compute.

        Priority 1:
        If an explicit managed identity is requested, use ManagedIdentityCredential instead.

        Priority 2:
        Perform service principal (SP) authentication.

        SP uses the Azure Identity package for Python.
        The DefaultAzureCredential class looks for the following environment variables and uses their values
        when authenticating as the SP: AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_SECRET. See more at:
        1. https://learn.microsoft.com/en-us/entra/identity-platform/app-objects-and-service-principals?tabs=browser
        2. https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication?view=azureml-api-2&tabs=sdk
        3. https://devblogs.microsoft.com/azure-sdk/authentication-and-the-azure-sdk/
        """

        use_managed_identity = os.getenv("USE_MANAGED_IDENTITY", "false").lower() == "true"
        managed_identity_client_id = os.getenv("AZURE_MANAGED_IDENTITY_CLIENT_ID", "").strip()

        if use_managed_identity:
            if managed_identity_client_id:
                logging.getLogger(__name__).info("Using user-assigned managed identity")
                return ManagedIdentityCredential(client_id=managed_identity_client_id)

            logging.getLogger(__name__).info("Using system-assigned managed identity")
            return ManagedIdentityCredential()

        logging.getLogger(__name__).info(f"Loading Azure configuration from {project_paths.AZURE_SETTINGS_YAML}")
        azure_config = cls.get_azure_config_from_yaml(project_paths.AZURE_SETTINGS_YAML)

        logging.getLogger(__name__).info(f"Loading Azure secrets from {project_paths.PRIVATE_AZURE_SECRETS_YAML}")
        azure_secrets = cls.get_azure_config_from_yaml(project_paths.PRIVATE_AZURE_SECRETS_YAML)

        # set environment variables needed to perform Azure authentication using DefaultAzureCredential
        logging.getLogger(__name__).info("Setting environment variables AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET")
        os.environ["AZURE_TENANT_ID"] = azure_config.AZURE_TENANT_ID
        os.environ["AZURE_CLIENT_ID"] = azure_config.AZURE_CLIENT_ID
        os.environ["AZURE_CLIENT_SECRET"] = azure_secrets.AZURE_CLIENT_SECRET

        # The DefaultAzureCredential tries to infer what environment is being used and uses the most appropriate credential.
        # Env variables set above are its 1st choice and will be used along with Azure Active Directory to authenticate the connection.
        # These environment variables define the service principal that was granted role-based permission to access data.
        #
        # NB: SP authentication is available for local development! Alternatively, one may want to use Managed Identity,
        # which would be the next choice for DefaultAzureCredential. Using Managed Identities eliminates the need to manage credentials,
        # but this option is apparently available only for Azure machines!

        return DefaultAzureCredential()
