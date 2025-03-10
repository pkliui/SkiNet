"""Tests for the AzureSetup class in SkiNet.Azure.azure_setup"""

import os
import pytest
import yaml
from pathlib import Path

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

from SkiNet.Azure.azure_setup import AzureSetup
from SkiNet.Utils import project_paths


"""------------------------------------------------------------------FIXTURES---------------------------------------------------------------"""

@pytest.fixture
def azure_settings_yaml(tmp_path):
    """
    Create a temporary YAML file with valid Azure configuration data as in SkiNet/SkiNet/azure_settings.yaml
    :return file_path, config: Path to the YAML file and the configuration dictionary
    """
    config = {
        "AZURE_TENANT_ID": "tenant_id_value",
        "AZURE_CLIENT_ID": "client_id_value",
        "SUBSCRIPTION_ID": "subscription_id_value",
        "RESOURCE_GROUP": "resource_group_value",
        "WORKSPACE_NAME": "workspace_name_value",
        "DATASTORE_NAME": "datastore_name_value",
        "PATH_ON_DATASTORE": "path_on_datastore_value"
    }
    file_path = tmp_path / "azure_settings.yaml"
    with file_path.open("w") as f:
        yaml.dump(config, f)
    return file_path, config

@pytest.fixture
def private_azure_secrets_yaml(tmp_path):
    """
    Create a temporary YAML file with user-specific Azure secrets.
    :return file_path, config: Path to the YAML file and the configuration dictionary
    """
    config = {
        "AZURE_CLIENT_SECRET": "secret_client_secret_value"
    }
    file_path = tmp_path / "private_azure_secrets.yaml"
    with file_path.open("w") as f:
        yaml.dump(config, f)
    return file_path, config

@pytest.fixture
def private_azure_secrets_yaml_missing_key(tmp_path):
    """
    Create a temporary YAML file with a missing key (AZURE_CLIENT_SECRET is required and missing)
    :return file_path, config: Path to the YAML file and the configuration dictionary
    """
    config = {
        "AZURE_CLIENT_SECRET": "",
    }
    file_path = tmp_path / "private_azure_secrets.yaml"
    with file_path.open("w") as f:
        yaml.dump(config, f)
    return file_path, config

"""------------------------------------------------------------------TESTS for get_azure_config_from_yaml---------------------------------------------------------------"""

def test_get_azure_config_from_yaml(azure_settings_yaml):
    """
    Test that config_from_yaml correctly loads a valid YAML configuration
    :param azure_settings_yaml: Fixture that creates a temporary YAML file with valid Azure configuration data
    """
    file_path, expected_config = azure_settings_yaml
    azure_config = AzureSetup.get_azure_config_from_yaml(file_path)
    
    for key, value in expected_config.items():
        assert getattr(azure_config, key) == value

def test_get_azure_config_from_yaml_file_not_found(tmp_path):
    """
    Test that get_azure_config_from_yaml raises a ValueError (raised in get_config_from_yaml) if the file does not exist
    """
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ValueError, match=f"Config file specified in {non_existent_file} not found"):
        AzureSetup.get_azure_config_from_yaml(non_existent_file)


"""------------------------------------------------------------------TESTS for perform_azure_authenticatio---------------------------------------------------------------"""

def test_service_principal_authentication_missing_key(monkeypatch, azure_settings_yaml, private_azure_secrets_yaml_missing_key):
    """
    Test that service_principal_authentication returns InteractiveBrowserCredential if a required key does not exist in provided YAML file
    """

    #Extract the paths to yaml files and their contents
    azure_config_path, _ = azure_settings_yaml
    secrets_path, secrets_config_dict = private_azure_secrets_yaml_missing_key

    # Monkeypatch the project_paths to point to YAML files as needed for service_principal_authentication function
    monkeypatch.setattr(project_paths, "AZURE_SETTINGS_YAML", azure_config_path)
    monkeypatch.setattr(project_paths, "PRIVATE_AZURE_SECRETS_YAML", secrets_path)
    
    # Clear the environment variables
    for var in ["AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"]:
        os.environ.pop(var, None)
    
    # Perform the authentication and do the checks
    credential = AzureSetup.service_principal_authentication()
    
    assert isinstance(credential, InteractiveBrowserCredential)


def test_service_principal_authentication_success(monkeypatch, azure_settings_yaml, private_azure_secrets_yaml):
    """
    Test that service_principal_authentication loads configurations from YAML files and sets the environment variables.
    """
    # Extract the paths to yaml files and their contents
    azure_config_path, azure_config_dict = azure_settings_yaml
    secrets_path, secrets_config_dict = private_azure_secrets_yaml

    # Monkeypatch the project_paths to point to YAML files as needed for service_principal_authentication function
    monkeypatch.setattr(project_paths, "AZURE_SETTINGS_YAML", azure_config_path)
    monkeypatch.setattr(project_paths, "PRIVATE_AZURE_SECRETS_YAML", secrets_path)
    
    # Clear the environment variables
    for var in ["AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"]:
        os.environ.pop(var, None)
    
    # Perform the authentication and do the checks
    AzureSetup.service_principal_authentication()

    assert os.environ["AZURE_TENANT_ID"] == azure_config_dict["AZURE_TENANT_ID"]
    assert os.environ["AZURE_CLIENT_ID"] == azure_config_dict["AZURE_CLIENT_ID"]
    assert os.environ["AZURE_CLIENT_SECRET"] == secrets_config_dict["AZURE_CLIENT_SECRET"]