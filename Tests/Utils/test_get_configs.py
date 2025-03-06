import pytest
import yaml
from pathlib import Path

from SkiNet.Utils.get_configs import get_config_from_yaml

"""------------------------------------------------------------------FIXTURES---------------------------------------------------------------"""

@pytest.fixture
def mimicked_yaml_file(tmp_path):
    """
    Fixture to mimick a YAML file and return its path and expected content
    """
    config = {
        "key1": "value1",
        "key2": 123,
        "nested": {"subkey": "subvalue"}
    }
    file_path = tmp_path / "config.yaml"
    with file_path.open("w") as f:
        yaml.dump(config, f)
    return file_path, config


@pytest.fixture
def mimicked_yaml_file_empty(tmp_path):
    """
    Fixture to create an empty YAML file and return its path
    """
    file_path = tmp_path / "empty.yaml"
    file_path.write_text("")
    return file_path

"""------------------------------------------------------------------TESTS for get_config_from_yaml---------------------------------------------------------------"""

def test_get_config_from_yaml_valid_content(mimicked_yaml_file):
    """
    Test reading a YAML file with a valid content
    """
    file_path, expected_config = mimicked_yaml_file
    loaded_config = get_config_from_yaml(file_path)
    assert loaded_config == expected_config

def test_get_config_from_yaml_empty(mimicked_yaml_file_empty):
    """
    Test reading an empty YAML file
    """
    file_path = mimicked_yaml_file_empty
    loaded_config = get_config_from_yaml(file_path)
    assert loaded_config == {}