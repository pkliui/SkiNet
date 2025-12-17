import logging
import os
from pathlib import Path
from typing import Any, Union

import param
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azureml.fsspec import AzureMachineLearningFileSystem

from SkiNet.Utils import project_paths
from SkiNet.Utils.get_configs import get_config_from_yaml


class AzureSetup(param.Parameterized):
    """
    Parametrized class used to set up Azure:
    - Load Azure configuration from a YAML file
    - Perform service principal authentication for e.g. data access in Azure ML
    """
    AZURE_TENANT_ID: str = param.String(doc="The tenant ID returned when you created the service principal")
    AZURE_CLIENT_ID: str = param.String(doc="The client ID returned when you created the service principal")
    AZURE_CLIENT_SECRET: str = param.String(doc="The password/credential generated for the service principal")
    SUBSCRIPTION_ID: str = param.String(doc="Azure subscription ID")
    RESOURCE_GROUP: str = param.String(doc="Resource group that contains the current workspace")
    WORKSPACE_NAME: str = param.String(doc="Name of the workspace")
    DATASTORE_NAME: str = param.String(doc="Name of the datastore containing data")
    PATH_ON_DATASTORE = param.Dict(doc="Dict mapping dataset names to paths on the datastore")

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

    @classmethod
    def get_azure_config_from_yaml(cls, path_to_yaml: Path) -> "AzureSetup":
        """
        Load Azure configuration from a YAML file
        :param path_to_yaml: Path to a YAML file with Azure configuration
        :return: Configured instance of AzureSetup
        """
        return cls(**get_config_from_yaml(path_to_yaml))

    @classmethod
    def get_azureml_filesystem(cls, dataset_name: str) -> AzureMachineLearningFileSystem:
        """
        Get AzureMachineLearningFileSystem

        :param dataset_name: Relative path to data on the datastore, w.r.t. to the datastore's root
        :return fs: AzureMachineLearningFileSystem object

        Example if data are under azure_uri:
            azure_uri = f"azureml://subscriptions/{azure_config.SUBSCRIPTION_ID}/resourcegroups/
            {azure_config.RESOURCE_GROUP}/workspaces/{azure_config.WORKSPACE_NAME}/datastores/{azure_config.DATASTORE_NAME}/paths/{path}/"
            In this case, "dataset_name" corresponds to "path"
        """
        azure_config = cls.get_azure_config_from_yaml(project_paths.AZURE_SETTINGS_YAML)

        # get path to data on the datastore w.r.t. its root
        if dataset_name not in azure_config.PATH_ON_DATASTORE:
            raise ValueError(f"Dataset name '{dataset_name}' not found in azure_config.PATH_ON_DATASTORE!"
                             f"Available options: {list(azure_config.PATH_ON_DATASTORE.keys())}")
        path = azure_config.PATH_ON_DATASTORE[dataset_name]

        azure_uri = f"azureml://subscriptions/{azure_config.SUBSCRIPTION_ID}/resourcegroups/{azure_config.RESOURCE_GROUP}/workspaces/{azure_config.WORKSPACE_NAME}/datastores/{azure_config.DATASTORE_NAME}/paths/{path}/"  # noqa: E501
        fs = AzureMachineLearningFileSystem(azure_uri)
        return fs

    @classmethod
    def service_principal_authentication(cls) -> Union[DefaultAzureCredential, InteractiveBrowserCredential]:
        """
        Perform service principal (SP) authentication. SP uses the Azure Identity package for Python.
        The DefaultAzureCredential class looks for the following environment variables and uses their values
        when authenticating as the SP: AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_SECRET. See more at:
        1. https://learn.microsoft.com/en-us/entra/identity-platform/app-objects-and-service-principals?tabs=browser
        2. https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication?view=azureml-api-2&tabs=sdk
        3. https://devblogs.microsoft.com/azure-sdk/authentication-and-the-azure-sdk/
        """
        logging.getLogger(__name__).info(f"Loading Azure configuration from {project_paths.AZURE_SETTINGS_YAML}")
        azure_config = cls.get_azure_config_from_yaml(project_paths.AZURE_SETTINGS_YAML)

        logging.getLogger(__name__).info(f"Loading Azure secrets from {project_paths.PRIVATE_AZURE_SECRETS_YAML}")
        azure_secrets = cls.get_azure_config_from_yaml(project_paths.PRIVATE_AZURE_SECRETS_YAML)

        # if some credentials are missing, use the InteractiveBrowserCredential
        if (not azure_config.AZURE_TENANT_ID) or (not azure_config.AZURE_CLIENT_ID) or (not azure_secrets.AZURE_CLIENT_SECRET):
            if not azure_config.AZURE_TENANT_ID:
                logging.getLogger(__name__).warning(f"Missing AZURE_TENANT_ID key in Azure config file {project_paths.AZURE_SETTINGS_YAML}")
            if not azure_config.AZURE_CLIENT_ID:
                logging.getLogger(__name__).warning(f"Missing AZURE_CLIENT_ID key in Azure config file {project_paths.AZURE_SETTINGS_YAML}")
            if not azure_secrets.AZURE_CLIENT_SECRET:
                logging.getLogger(__name__).warning(f"Missing AZURE_CLIENT_SECRET key in Azure config file {project_paths.PRIVATE_AZURE_SECRETS_YAML}")
            logging.getLogger(__name__).info("Using InteractiveBrowserCredential to authenticate with Azure")

            return InteractiveBrowserCredential()

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
