"""Tests for the AzureSetup class in SkiNet.Azure.azure_setup"""

import os
from pathlib import Path
from typing import Generator, Tuple, Type

import pytest
import yaml
from azure.identity import InteractiveBrowserCredential
from azureml.fsspec import AzureMachineLearningFileSystem

from SkiNet.Azure.azure_setup import AzureSetup
from SkiNet.Utils import project_paths

"""------------------------------------------------------------------FIXTURES---------------------------------------------------------------"""

@pytest.fixture
def azure_settings_yaml(tmp_path: Path) -> Tuple[Path, dict]:
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
        "PATH_ON_DATASTORE": {"PH22": "path_on_datastore_value"}
    }
    file_path = tmp_path / "azure_settings.yaml"
    with file_path.open("w") as f:
        yaml.dump(config, f)
    return file_path, config


@pytest.fixture
def private_azure_secrets_yaml(tmp_path: Path) -> Tuple[Path, dict]:
    """
    Create a temporary YAML file with user-specific Azure secrets.
    :return file_path, config: Path to the YAML file and the configuration dictionary
    """
    config = {"AZURE_CLIENT_SECRET": "secret_client_secret_value"}
    file_path = tmp_path / "private_azure_secrets.yaml"
    with file_path.open("w") as f:
        yaml.dump(config, f)
    return file_path, config


@pytest.fixture
def private_azure_secrets_yaml_missing_key(tmp_path: Path) -> Tuple[Path, dict]:
    """
    Create a temporary YAML file with a missing key (AZURE_CLIENT_SECRET is required and missing)
    :return file_path, config: Path to the YAML file and the configuration dictionary
    """
    config = {"AZURE_CLIENT_SECRET": ""}
    file_path = tmp_path / "private_azure_secrets.yaml"
    with file_path.open("w") as f:
        yaml.dump(config, f)
    return file_path, config


@pytest.fixture
def mock_default_azure_credential(monkeypatch: pytest.MonkeyPatch) -> Type:
    """
    Mock SkiNet.Azure.azure_setup.DefaultAzureCredential
    """
    class DummyCredential:
        pass

    monkeypatch.setattr("SkiNet.Azure.azure_setup.DefaultAzureCredential", DummyCredential)
    return DummyCredential


@pytest.fixture
def clear_azure_env_vars() -> Generator[None, None, None]:
    """
    Clear environment variables for Azure credentials, restoring old values after the test
    """
    old_vars = {var: os.environ.get(var) for var in ["AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"]}
    for var in old_vars:
        os.environ.pop(var, None)
    yield
    for var, value in old_vars.items():
        if value is not None:
            os.environ[var] = value


"""-----------------------------------------------TESTS for get_azure_config_from_yaml---------------------------------------------------------------"""

def test_get_azure_config_from_yaml(azure_settings_yaml: Tuple[Path, dict]) -> None:
    """
    Test that get_azure_config_from_yaml correctly loads a valid YAML configuration
    :param azure_settings_yaml: Fixture that creates a temporary YAML file with valid Azure configuration data
    """
    file_path, expected_config = azure_settings_yaml
    azure_config = AzureSetup.get_azure_config_from_yaml(file_path)
    for key, value in expected_config.items():
        assert getattr(azure_config, key) == value


def test_get_azure_config_from_yaml_file_not_found(tmp_path: Path) -> None:
    """
    Test that get_azure_config_from_yaml raises a ValueError if the file does not exist
    """
    non_existent_file = tmp_path / "non_existent.yaml"
    with pytest.raises(ValueError, match=f"Config file specified in {non_existent_file} not found"):
        _ = AzureSetup.get_azure_config_from_yaml(non_existent_file)


"""----------------------------------------------------TESTS for perform_azure_authentication---------------------------------------------------------------"""

def test_service_principal_authentication_missing_key(monkeypatch: pytest.MonkeyPatch,
                                                      azure_settings_yaml: Tuple[Path, dict],
                                                      private_azure_secrets_yaml_missing_key: Tuple[Path, dict],
                                                      clear_azure_env_vars: None) -> None:
    """
    Test that service_principal_authentication returns InteractiveBrowserCredential
    if a required key does not exist in provided YAML file
    """
    _ = clear_azure_env_vars
    azure_config_path, _ = azure_settings_yaml
    secrets_path, _ = private_azure_secrets_yaml_missing_key
    monkeypatch.setattr(project_paths, "AZURE_SETTINGS_YAML", azure_config_path)
    monkeypatch.setattr(project_paths, "PRIVATE_AZURE_SECRETS_YAML", secrets_path)

    credential = AzureSetup.service_principal_authentication()
    assert isinstance(credential, InteractiveBrowserCredential)
    assert "AZURE_CLIENT_SECRET" not in os.environ


def test_service_principal_authentication_success(monkeypatch: pytest.MonkeyPatch,
                                                  azure_settings_yaml: Tuple[Path, dict],
                                                  private_azure_secrets_yaml: Tuple[Path, dict],
                                                  mock_default_azure_credential: Type,
                                                  clear_azure_env_vars: None) -> None:
    """
    Test that service_principal_authentication loads configurations from YAML files
    and sets the environment variables
    """
    _ = clear_azure_env_vars
    azure_config_path, azure_config_dict = azure_settings_yaml
    secrets_path, secrets_config_dict = private_azure_secrets_yaml
    monkeypatch.setattr(project_paths, "AZURE_SETTINGS_YAML", azure_config_path)
    monkeypatch.setattr(project_paths, "PRIVATE_AZURE_SECRETS_YAML", secrets_path)

    credential = AzureSetup.service_principal_authentication()
    assert isinstance(credential, mock_default_azure_credential)
    assert os.environ["AZURE_TENANT_ID"] == azure_config_dict["AZURE_TENANT_ID"]
    assert os.environ["AZURE_CLIENT_ID"] == azure_config_dict["AZURE_CLIENT_ID"]
    assert os.environ["AZURE_CLIENT_SECRET"] == secrets_config_dict["AZURE_CLIENT_SECRET"]


"""---------------------------------------------------TESTS for get_azureml_filesystem---------------------------------------------------------------"""

def test_get_azureml_filesystem_success(monkeypatch: pytest.MonkeyPatch,
                                        azure_settings_yaml: Tuple[Path, dict],
                                        mock_default_azure_credential: Type) -> None:
    """
    Test that we can get an AzureMachineLearningFileSystem object
    given a relative path to data on the datastore
    """
    _ = mock_default_azure_credential
    azure_settings_yaml_path, _ = azure_settings_yaml
    monkeypatch.setattr(project_paths, "AZURE_SETTINGS_YAML", azure_settings_yaml_path)
    _ = AzureSetup.service_principal_authentication()
    fs = AzureSetup.get_azureml_filesystem("PH22")
    assert isinstance(fs, AzureMachineLearningFileSystem)


def test_get_azureml_filesystem_dataset_not_found(monkeypatch: pytest.MonkeyPatch,
                                                  azure_settings_yaml: Tuple[Path, dict]) -> None:
    """
    Test raising ValueError given a non-existing relative path to data on the datastore
    """
    file_path, config = azure_settings_yaml
    monkeypatch.setattr(project_paths, "AZURE_SETTINGS_YAML", file_path)
    config["PATH_ON_DATASTORE"] = {"not_ph22": "non_existing_path"}
    with file_path.open("w") as f:
        yaml.dump(config, f)
    with pytest.raises(ValueError):
        AzureSetup.get_azureml_filesystem("PH22")
