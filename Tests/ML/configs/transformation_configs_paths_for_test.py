"""
Construct transformation configs for their use in unit tests
"""
from SkiNet.ML.configs import transformations_config
from SkiNet.Utils.project_paths_tests import TRANSFORMATION_CONFIGS_YAML_PATH 

# default configuration 
# to be used in test_transform_data.py/test_make_transform_from_config_PIL_image
config_test_transform_data = transformations_config.get_default_config()

# configuration that uses YAML overrriding defaults 
# to be used in test_transform_data.py/test_make_transform_from_configYAML_PIL_image
config_yaml_test_transform_data = transformations_config.get_default_config()
config_yaml_test_transform_data.merge_from_file(TRANSFORMATION_CONFIGS_YAML_PATH) # override from YAML
config_yaml_test_transform_data.freeze() #  to prevent further modification
