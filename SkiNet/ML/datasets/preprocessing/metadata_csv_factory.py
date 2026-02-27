import argparse
import logging
from abc import ABC, abstractmethod
from typing import Union

from SkiNet.ML.configs.datasets.dataset_keys import DatasetKey
from SkiNet.ML.datasets.preprocessing.base_csv_builder import AzureCSVBuilder, LocalCSVBuilder
from SkiNet.ML.datasets.preprocessing.ph2_csv_builder import PH2AzureCSVBuilder, PH2LocalCSVBuilder


class MetadataFactory(ABC):
    """
    Abstract base class for creating metadata in different environments (local, Azure).
    """

    @abstractmethod
    def get_local_csv_builder(self, arg: argparse.Namespace) -> LocalCSVBuilder:
        """
        Create a CSV builder instance for the local environment.
        """
        pass

    @abstractmethod
    def get_azure_csv_builder(self) -> AzureCSVBuilder:
        """
        Create a CSV builder instance for the Azure environment.
        """
        pass


class PH2MetadataFactory(MetadataFactory):
    """
    Factory class for creating PH2 dataset metadata in local and Azure environments.
    """

    def get_local_csv_builder(self, arg: argparse.Namespace) -> LocalCSVBuilder:
        """
        Create a LocalCSVBuilder instance for the PH2 dataset.
        """
        return PH2LocalCSVBuilder(arg)

    def get_azure_csv_builder(self) -> AzureCSVBuilder:
        """
        Create an AzureCSVBuilder instance for the PH2 dataset.
        """
        return PH2AzureCSVBuilder()


def get_factory(dataset_key: DatasetKey) -> MetadataFactory:
    """
    Get a specific factory for creating metadata based on the dataset key.
    This function can be extended to return different factories for different datasets.
    """

    factories = {
        DatasetKey.PH2: PH2MetadataFactory()
    }

    if dataset_key in factories:
        return factories[dataset_key]
    else:
        logging.getLogger(__name__).error(f"No factory found for dataset key: {dataset_key}")
        raise ValueError(f"No factory found for dataset key: {dataset_key}")


def main(args: argparse.Namespace) -> None:
    """
    Main function to create metadata CSV for the specified dataset.
    This function determines the dataset key as per the command-line arguments, checks for valid argument combinations,
    and uses the appropriate factory and builder to create the metadata CSV.
    """
    # Convert string to enum, i.e. "PH2" -> DatasetKey.PH2
    try:
        dataset_key = DatasetKey[args.dataset_key.upper()]
    except KeyError:
        raise ValueError(f"Unknown dataset key: {args.dataset_key}. Valid options: {[k.name for k in DatasetKey]}")

    if args.azure_data and args.local_data_root is not None:
        raise ValueError("Do not provide --local-data-root when using --azure-data.")

    factory = get_factory(dataset_key=dataset_key)

    builder: Union[AzureCSVBuilder, LocalCSVBuilder]
    if args.azure_data:
        builder = factory.get_azure_csv_builder()
    else:
        builder = factory.get_local_csv_builder(args)
    builder.create_metadata_csv()


if __name__ == "__main__":
    """
    Entry point for creating PH2 metadata.

    Example for local data:
    ```
    python metadata_csv_factory.py --dataset_key="PH2" --local_data_root="path/to/local/data/PH2folder"
    ```

    Example for Azure data:
    ```
    python metadata_csv_factory.py --dataset_key="PH2" --azure_data,
    ```

    where "PH2" is a valid dataset key from the DatasetKey Enum.
    For local data, the local_data_root should point to the directory containing the PH2 data samples.
    For data on Azure, the script will look for the data location specified in the YAML config file.
    """
    parser = argparse.ArgumentParser(description="Create metadata CSV for SkiNet.")
    parser.add_argument("--dataset-key",
                        type=str,
                        required=True,
                        help="Dataset identifier key. Should be one specific key from DatasetKey, uniquely "
                        "identifying the dataset for which you want to create metadata. For example, use 'PH2' for the PH2 dataset.")
    parser.add_argument("--azure-data",
                        action='store_true',
                        required=False,
                        help="If True (added as --azure-data), local_data_root should not be provided and the script will look for data in Azure. "
                        "If False, local_data_root must be provided and the script will look for data on the local file system.")
    parser.add_argument("--local-data-root",
                        type=str,
                        required=False,
                        help="The root path to data on the local file system. The path should point to a directory that contains folders"
                        " with samples of data uniquely identifiable by their ID. Only used when no --azure-data flag is set.")
    args = parser.parse_args()

    main(args)
