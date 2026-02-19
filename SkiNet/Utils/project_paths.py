"""
Specifies paths used in SkiNet project

The repository's root directory is SkiNet/ and is structured as follows:
SkiNet               - the repository's root directory
|__SkiNet            - the projects's root directory
|  |__Azure          - contains Azure-related code
|  |__ML             - contains machine learning-related code
|  |__Plotting       - code for plotting
|  |__Utils          - contains various utility code
|
|__Tests             - contains tests for the project
|__docs              - project's documentation
|__environment.yaml  - conda environment file

"""
from pathlib import Path


def get_repo_root_directory() -> Path:
    """
    Return the full path to the repository's root directory
    """
    current = Path(__file__)
    root = current.parent.parent.parent
    return root


# ________________REPO_PATHS_____________
# Repo root
REPO_ROOT = get_repo_root_directory()
# Name of the project
SKINET_PROJECT_NAME = "SkiNet"
# Root directory of the project, i.e. SkiNet/SkiNet
SKINET_ROOT_DIR = get_repo_root_directory() / SKINET_PROJECT_NAME

# ________________TESTS__________________
# Name and path to Tests directory
TEST_DIR_NAME = "Tests"
TESTS_DIR = get_repo_root_directory() / TEST_DIR_NAME

# ______________SKINET/AZURE_____________
# Name and path to a directory containing Azure-related code
AZURE_DIR_NAME = "Azure"
AZURE_DIR = SKINET_ROOT_DIR / AZURE_DIR_NAME

# Name and path to a file containing settings for Azure
AZURE_SETTINGS_YAML_NAME = "azure_settings.yaml"
AZURE_SETTINGS_YAML = AZURE_DIR / AZURE_SETTINGS_YAML_NAME

# Name and path to a file keeping Azure secrets - must NOT be version-controlled!!!
PRIVATE_AZURE_SECRETS_YAML_NAME = "PrivateAzureSecrets.yaml"
PRIVATE_AZURE_SECRETS_YAML = AZURE_DIR / PRIVATE_AZURE_SECRETS_YAML_NAME

# ______________Data-related_____________
# NB: For the dataset keys see SkiNet.ML.configs.datasets.dataset_keys.AzureDatasetKey class
# Name of the CSV file containing metadata for the PH2 dataset, how it is saved or read from the disk
PH2_CSV_NAME = "ph2_metadata.csv"
