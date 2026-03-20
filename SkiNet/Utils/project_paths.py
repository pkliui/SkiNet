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

# PH2 dataset

# NB: For the dataset keys see SkiNet.ML.configs.datasets.dataset_keys.DatasetKey class
# Name of the CSV file containing metadata for the PH2 dataset, how it is saved or read from the disk
PH2_CSV_NAME = "ph2_metadata.csv"

# Name of the TXT file containing the PH2 dataset's metadata as it is originally provided by the dataset authors
PH2_TXT_NAME = "PH2_dataset.txt"

# Local image and mask patterns
PH2_IMAGE_PATTERN_LOCAL = "**/*_Dermoscopic_Image/*.bmp"
PH2_MASK_PATTERN_LOCAL = "**/*_lesion/*.bmp"
PH2_IMAGE_PATTERN_AZURE = "**_Dermoscopic_Image/**.bmp"
PH2_MASK_PATTERN_AZURE = "**_lesion/**.bmp"


# ______________Data-related for Azure machine_____________
# Path to a directory on Azure where data from Azure Blob Container is mounted
AZURE_MOUNT_PATH: Path = Path("/mnt/azure_blob_data/")

# Path to blobfuse2 configuration file
BLOBFUSE2_CONFIG_PATH = AZURE_DIR / "blobfuse2.yaml"
