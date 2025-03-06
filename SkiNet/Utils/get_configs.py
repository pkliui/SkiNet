import yaml 
from typing import Any, Dict, Optional, cast, List, Union
from pathlib import Path


def get_config_from_yaml(path_to_yaml: Path)-> Dict:
    """
    Read a YAML configuration file and return its contents as a dictionary.

    :param path_to_yaml: The file path to the YAML configuration file
    :return: A dictionary containing the configuration from the YAML file.
    """
    with open(path_to_yaml, 'r') as file:
        yaml_data = yaml.safe_load(file)
        if yaml_data:
            return yaml_data
        return dict()
