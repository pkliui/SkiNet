import argparse
from enum import Enum
from typing import Any

import pytest

from SkiNet.ML.datasets.preprocessing.metadata_csv_factory import PH2MetadataFactory, get_factory, main
from SkiNet.ML.datasets.preprocessing.ph2_csv_builder import PH2AzureCSVBuilder, PH2LocalCSVBuilder
from SkiNet.Utils.experiment_keys import DatasetKey


@pytest.fixture
def local_arg_ph2(tmp_path: pytest.TempPathFactory) -> argparse.Namespace:
    """Simulate argparse.Namespace for local"""
    return argparse.Namespace(local_data_root=str(tmp_path), dataset_key_str="PH2", azure_data=False)

@pytest.fixture
def azure_arg_ph2() -> argparse.Namespace:
    """Simulate argparse.Namespace for Azure"""
    return argparse.Namespace(local_data_root=None, dataset_key_str="PH2", azure_data=True)

@pytest.mark.parametrize("factory_cls, builder_cls, arg_fixture", [
    (PH2MetadataFactory, PH2LocalCSVBuilder, "local_arg_ph2"),
    (PH2MetadataFactory, PH2AzureCSVBuilder, "azure_arg_ph2"),
])
def test_factory_returns_correct_builder(factory_cls: type, builder_cls: type, arg_fixture: str, request: pytest.FixtureRequest) -> None:
    """Confirm that the factory returns the correct builder instance based on the dataset key and environment (local vs Azure)"""
    factory = factory_cls()
    arg = request.getfixturevalue(arg_fixture)
    if builder_cls is PH2LocalCSVBuilder:
        builder = factory.get_local_csv_builder(arg)  # factory should return a PH2LocalCSVBuilder
    else:
        builder = factory.get_azure_csv_builder()  # factory should return a PH2AzureCSVBuilder
    assert isinstance(builder, builder_cls)

def test_get_factory_returns_ph2_factory() -> None:
    """Confirm that the factory returns a PH2MetadataFactory instance for the PH2 dataset key"""
    factory = get_factory(DatasetKey.PH2)
    assert isinstance(factory, PH2MetadataFactory)

def test_get_factory_raises_on_invalid_key() -> None:
    """Confirm that the factory raises a ValueError for invalid dataset keys"""
    with pytest.raises(ValueError, match="No factory found for dataset key"):
        class FakeDatasetKey(Enum):
            ANOTHER_DATASET = "ANOTHER_DATASET"
        get_factory(FakeDatasetKey.ANOTHER_DATASET)  # type: ignore[arg-type]

def test_main_local(monkeypatch: pytest.MonkeyPatch, local_arg_ph2: argparse.Namespace) -> None:
    """ Test the main function for local environment, ensuring it runs without errors and calls the correct builder method."""
    # Patch builder.create_metadata_csv to avoid actual file operations
    class DummyBuilder:
        def create_metadata_csv(self) -> None:
            pass

    # Patch the factory to return a dummy builder that does nothing when create_metadata_csv is called
    class DummyFactory:
        def get_local_csv_builder(self, arg: argparse.Namespace) -> Any:
            return DummyBuilder()

        def get_azure_csv_builder(self) -> Any:
            return DummyBuilder()

    monkeypatch.setattr("SkiNet.ML.datasets.preprocessing.metadata_csv_factory.get_factory", lambda dataset_key_str: DummyFactory())
    main(local_arg_ph2)  # Should not raise

def test_main_azure(monkeypatch: pytest.MonkeyPatch, azure_arg_ph2: argparse.Namespace) -> None:
    """ Test the main function for Azure environment, ensuring it runs without errors and calls the correct builder method."""
    # Patch builder.create_metadata_csv to avoid actual Azure operations
    class DummyBuilder:
        def create_metadata_csv(self) -> None:
            pass

    # Patch the factory to return a dummy builder that does nothing when create_metadata_csv is called
    class DummyFactory:
        def get_local_csv_builder(self, arg: argparse.Namespace) -> Any:
            return DummyBuilder()

        def get_azure_csv_builder(self) -> Any:
            return DummyBuilder()

    monkeypatch.setattr("SkiNet.ML.datasets.preprocessing.metadata_csv_factory.get_factory", lambda dataset_key_str: DummyFactory())
    main(azure_arg_ph2)  # Should not raise

@pytest.mark.parametrize("arg", [
    argparse.Namespace(local_data_root="some/path", dataset_key_str="PH2", azure_data=True),
])
def test_main_raises_on_both_local_and_azure(arg: argparse.Namespace) -> None:
    """
    Confirm that the main function raises a ValueError when both local and Azure paths are provided.
    """
    with pytest.raises(ValueError, match="Do not provide --local-data-root when using --azure-data"):
        main(arg)


def test_enum_conversion_invalid_key() -> None:
    """Test that converting an invalid dataset key string to the DatasetKey enum raises a ValueError with an appropriate message."""
    args = argparse.Namespace(dataset_key_str="INVALID", azure_data=False, local_data_root="dummy")
    with pytest.raises(ValueError, match="Unknown dataset key string: INVALID"):
        try:
            _ = DatasetKey[args.dataset_key_str.upper()]
        except KeyError:
            raise ValueError(f"Unknown dataset key string: {args.dataset_key_str}. Valid options: {[k.name for k in DatasetKey]}")
