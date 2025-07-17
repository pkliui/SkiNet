
from pathlib import Path
from yacs.config import CfgNode

def load_transform_config(transform_config_yaml_path: Path) -> CfgNode:
    """
    Loads transformation configs from yaml files

    :param transform_config_yaml_path: Path to the yaml file with transformation configs for a particular dataset
    """
    # import default config
    from SkiNet.ML.configs import transformations_config
    config = transformations_config.get_default_config()

    # import yaml settings
    config.merge_from_file(transform_config_yaml_path) # override from YAML
    config.freeze() #  to prevent further modification
    return config