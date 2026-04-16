from types import SimpleNamespace

import pandas as pd
import pytest
from typing import Any, cast
from pathlib import Path

from SkiNet.ML.datasets.dataset_factory import (
    SegmentationDatasetFactory,
    create_segmentation_datasets_from_config,
    _get_dataset_factory,
)
from SkiNet.Utils.data.split_data import DataFrameSplits, SplitConfig
from SkiNet.Utils.experiment_keys import ExperimentType
from SkiNet.ML.utils.model_utils import MLWorkflowState
from SkiNet.ML.configs.experiment_config import ExperimentConfig


class DummyDatasetConfig:
    """
    Dummy dataset config to provide necessary attributes and methods for testing the dataset factory.
    """

    def __init__(self, metadata: pd.DataFrame, split_config: SplitConfig, data_root: Path) -> None:
        self.metadata = metadata
        self.split_config = split_config
        self.data_root = data_root

    def get_split_config(self) -> SplitConfig:
        # In a real implementation, this involves more complex logic to determine the split config,
        # here we just return the provided split_config for testing purposes.
        return self.split_config


def _make_config(experiment_type: Any = ExperimentType.SEGMENTATION) -> ExperimentConfig:
    """
    Helper to create a dummy ExperimentConfig for segmentation experiments,
    with a simple metadata DataFrame and a SplitConfig.
    """
    metadata_df = pd.DataFrame({"sampleid": ["a", "a", "b", "b"]})
    split_config = SplitConfig(
        train_size=0.7,
        val_size=0.1,
        test_size=0.2,
        stratify_column="Clinical Diagnosis",
        random_seed=42,
    )
    data_root = Path("somepath")
    return cast(
        ExperimentConfig,
        SimpleNamespace(
            experiment_type=experiment_type,
            dataconfig=DummyDatasetConfig(metadata=metadata_df, split_config=split_config, data_root=data_root),
        ),
    )


def test_get_dataset_factory_returns_segmentation_factory() -> None:
    """
    Test that _get_dataset_factory returns a SegmentationDatasetFactory
    when the experiment type is SEGMENTATION.
    """
    config = _make_config()
    factory = _get_dataset_factory(config=config)
    assert isinstance(factory, SegmentationDatasetFactory)


def test_get_dataset_factory_raises_for_unsupported_experiment_type() -> None:
    """
    Test that _get_dataset_factory raises a ValueError for an unregistered experiment type.
    """
    config = _make_config(experiment_type="unsupported")
    with pytest.raises(ValueError, match="Unsupported experiment type"):
        _get_dataset_factory(config)


def test_segmentation_dataset_factory_creates_expected_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that SegmentationDatasetFactory.create_datasets() correctly orchestrates
    its dependencies and assembles the returned DatasetSplit.

    Replaces split_segmentation_metadata, get_transform_from_config, and
    SegmentationDataset with lightweight fakes to isolate the factory's own logic.

    Asserts that:
    - split_segmentation_metadata is called with the metadata and split_config
      taken directly from the provided ExperimentConfig (verified by identity).
    - get_transform_from_config is called with the same ExperimentConfig instance.
    - Exactly three SegmentationDataset instances are created, one per split.
    - Each dataset receives the correct dataframe, transform branch, and
      MLWorkflowState in the right combination and order.
    - The returned split carries the splits object produced by split_segmentation_metadata.
    """
    config = _make_config()
    factory = SegmentationDatasetFactory()

    train_df = pd.DataFrame({"sampleid": ["train"]})
    val_df = pd.DataFrame({"sampleid": ["val"]})
    test_df = pd.DataFrame({"sampleid": ["test"]})

    fake_splits = DataFrameSplits(train=train_df, val=val_df, test=test_df)

    def fake_split_segmentation_metadata(df: pd.DataFrame, split_config: SplitConfig) -> DataFrameSplits:
        assert df is config.dataconfig.metadata
        assert split_config is config.dataconfig.get_split_config()
        return fake_splits

    fake_transform = SimpleNamespace(train="train_tf", val="val_tf", test="test_tf")

    def fake_get_transform_from_config(cfg: object) -> SimpleNamespace:
        assert cfg is config
        return fake_transform

    created_datasets: list[dict[str, object]] = []

    class FakeSegmentationDataset:
        """
        Minimal stand-in for SegmentationDataset that records constructor arguments
        so assertions can verify the factory wired everything correctly.
        """

        def __init__(self, config: object, dataframe: pd.DataFrame, transform: object, mode: object) -> None:
            created_datasets.append({
                "config": config,
                "dataframe": dataframe,
                "transform": transform,
                "mode": mode,
            })

    monkeypatch.setattr("SkiNet.ML.datasets.dataset_factory.split_segmentation_metadata",
                        fake_split_segmentation_metadata)
    monkeypatch.setattr("SkiNet.ML.datasets.dataset_factory.get_transform_from_config", fake_get_transform_from_config)
    monkeypatch.setattr("SkiNet.ML.datasets.dataset_factory.SegmentationDataset", FakeSegmentationDataset)

    split = factory.create_datasets(config)

    assert split.splits is fake_splits
    assert len(created_datasets) == 3

    assert created_datasets[0]["dataframe"] is train_df
    assert created_datasets[0]["transform"] == "train_tf"
    assert created_datasets[0]["mode"] == MLWorkflowState.TRAIN

    assert created_datasets[1]["dataframe"] is val_df
    assert created_datasets[1]["transform"] == "val_tf"
    assert created_datasets[1]["mode"] == MLWorkflowState.VAL

    assert created_datasets[2]["dataframe"] is test_df
    assert created_datasets[2]["transform"] == "test_tf"
    assert created_datasets[2]["mode"] == MLWorkflowState.TEST


def test_create_segmentation_datasets_from_config_delegates_to_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Test that create_segmentation_datasets_from_config delegates to
    SegmentationDatasetFactory.create_datasets and returns its result unchanged.
    """
    config = _make_config()
    expected_split = SimpleNamespace(name="split")

    class FakeSegmentationDatasetFactory:
        def create_datasets(self, cfg: ExperimentConfig) -> object:
            assert cfg is config
            return expected_split

    monkeypatch.setattr(
        "SkiNet.ML.datasets.dataset_factory.SegmentationDatasetFactory",
        lambda: FakeSegmentationDatasetFactory(),
    )

    result = create_segmentation_datasets_from_config(config)
    assert result is expected_split
