from types import SimpleNamespace

import pandas as pd
import pytest
from typing import Any, cast


from SkiNet.ML.datasets.dataset_factory import (
    SegmentationDatasetFactory,
    create_pytorch_datasets_from_config,
    _get_dataset_factory,
)
from SkiNet.Utils.data.split_data import DataFrameSplits, SplitConfig
from SkiNet.Utils.experiment_keys import ExperimentType
from SkiNet.ML.utils.model_utils import MLWorkflowState
from SkiNet.ML.configs.experiment_config import ExperimentConfig


class DummyDatasetConfig:
    """
    Dummy dataset config class to provide necessary attributes and methods for testing the dataset factory.
    """

    def __init__(self, metadata: pd.DataFrame, split_config: SplitConfig) -> None:
        self.metadata = metadata
        self._split_config = split_config

    def get_split_config(self) -> SplitConfig:
        return self._split_config


def _make_config(experiment_type: Any = ExperimentType.SEGMENTATION) -> ExperimentConfig:
    """
    Helper function to create a dummy ExperimentConfig for segmentation experiments,
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
    return cast(
        ExperimentConfig,
        SimpleNamespace(
            experiment_type=experiment_type,
            dataconfig=DummyDatasetConfig(metadata=metadata_df, split_config=split_config),
        ),
    )


def test_get_dataset_factory_returns_segmentation_factory() -> None:
    """
    Test that _get_dataset_factory returns an instance of SegmentationDatasetFactory when the experiment type is SEGMENTATION.
    """
    # Arrange - config must have experiment_type set to SEGMENTATION for this test, which is the default in _make_config
    config = _make_config()
    # Act
    factory = _get_dataset_factory(config=config)

    assert isinstance(factory, SegmentationDatasetFactory)


def test_get_dataset_factory_raises_for_unsupported_experiment_type() -> None:
    """
    Test that _get_dataset_factory raises a ValueError when the experiment type is unsupported.
    """
    config = _make_config(experiment_type="unsupported")
    with pytest.raises(ValueError, match="Unsupported experiment type"):
        _get_dataset_factory(config)


def test_segmentation_dataset_factory_creates_expected_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that SegmentationDatasetFactory.create_datasets() correctly orchestrates its
    dependencies and assembles the returned DatasetBundle.

    The test replaces the three collaborators that the factory calls —
    split_segmentation_metadata, get_transform_from_config, and SegmentationDataset —
    with lightweight fakes, isolating the factory's own logic from filesystem access,
    real splits, and real transform pipelines.

    Specifically, the test asserts that:
    - split_segmentation_metadata is called with the metadata and split_config
      taken directly from the provided ExperimentConfig (verified by identity).
    - get_transform_from_config is called with the same ExperimentConfig instance.
    - Exactly three SegmentationDataset instances are created, one per split.
    - Each dataset receives the correct dataframe from the splits object (train/val/test),
      the corresponding transform branch (train/val/test), and the correct MLWorkflowState
      mode — in the right combination and order.
    - The returned bundle carries the splits object produced by split_segmentation_metadata.
    """
    # Arrange - config must have experiment_type set to SEGMENTATION for this test, which is the default in _make_config
    config = _make_config()
    factory = SegmentationDatasetFactory()

    # create simple DataFrames to represent the splits that would be returned by split_segmentation_metadata
    # based on the input metadata and split config in the config
    train_df_from_split_segmentation_metadata = pd.DataFrame({"sampleid": ["train"]})
    val_df_from_split_segmentation_metadata = pd.DataFrame({"sampleid": ["val"]})
    test_df_from_split_segmentation_metadata = pd.DataFrame({"sampleid": ["test"]})

    # stand-in for what split_segmentation_metadata would return based on the input metadata and split config in the config
    fake_splits = DataFrameSplits(train=train_df_from_split_segmentation_metadata,
                                  val=val_df_from_split_segmentation_metadata, test=test_df_from_split_segmentation_metadata)

    def fake_split_segmentation_metadata(df: pd.DataFrame, split_config: SplitConfig) -> DataFrameSplits:
        assert df is config.dataconfig.metadata
        assert split_config is config.dataconfig.get_split_config()
        return fake_splits

    # stand-in for what get_transform_from_config would return based on the input config
    # SimpleNamespace is enough because the factory never inspects or calls the transform.
    fake_transform = SimpleNamespace(train="train_tf", val="val_tf", test="test_tf")

    def fake_get_transform_from_config(cfg: object) -> SimpleNamespace:
        assert cfg is config
        return fake_transform

    # list to capture the datasets created by the factory for assertion later
    created_datasets: list[dict[str, object]] = []

    class FakeSegmentationDatasetAsInFactory:
        """
        Make a fake SegmentationDataset class that accepts the same constructor signature as in SegmentationDatasetFactory
        """

        def __init__(self, config: object, dataframe: pd.DataFrame, transform: object, mode: object) -> None:
            created_datasets.append(
                {
                    "config": config,
                    "dataframe": dataframe,
                    "transform": transform,
                    "mode": mode,
                }
            )
            self.config = config
            self.dataframe = dataframe
            self.transform = transform
            self.mode = mode

    # Path where split_segmentation_metadata and other modules where they are used, not where defined
    monkeypatch.setattr(
        "SkiNet.ML.datasets.dataset_factory.split_segmentation_metadata",
        fake_split_segmentation_metadata,
    )
    monkeypatch.setattr(
        "SkiNet.ML.datasets.dataset_factory.get_transform_from_config",
        fake_get_transform_from_config,
    )
    monkeypatch.setattr(
        "SkiNet.ML.datasets.dataset_factory.SegmentationDataset",
        FakeSegmentationDatasetAsInFactory,
    )

    # Act
    # Call the method under test, which will use the fakes and populate created_datasets
    bundle = factory.create_datasets(config)

    assert bundle.splits == fake_splits
    assert len(created_datasets) == 3

    # check if the exact same object in memory was passed to the SegmentationDataset constructor for each split
    assert created_datasets[0]["dataframe"] is train_df_from_split_segmentation_metadata
    # check the value of the string, memory identity is irrelevant for the transform since it's a string in this test
    assert created_datasets[0]["transform"] == "train_tf"
    assert created_datasets[0]["mode"] == MLWorkflowState.TRAIN

    assert created_datasets[1]["dataframe"] is val_df_from_split_segmentation_metadata
    assert created_datasets[1]["transform"] == "val_tf"
    assert created_datasets[1]["mode"] == MLWorkflowState.VAL

    assert created_datasets[2]["dataframe"] is test_df_from_split_segmentation_metadata
    assert created_datasets[2]["transform"] == "test_tf"
    assert created_datasets[2]["mode"] == MLWorkflowState.TEST


def test_create_pytorch_datasets_from_config_delegates_to_selected_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Test that create_pytorch_datasets_from_config correctly delegates to the dataset factory selected by _get_dataset_factory
    """
    config = _make_config()
    expected_bundle = SimpleNamespace(name="bundle")

    class FakeFactory:
        def create_datasets(self, cfg: ExperimentConfig) -> object:
            print("cfg.experiment_type:", cfg.experiment_type)
            print("ExperimentType.SEGMENTATION :", ExperimentType.SEGMENTATION)

            assert cfg.experiment_type == ExperimentType.SEGMENTATION
            return expected_bundle

    monkeypatch.setattr(
        "SkiNet.ML.datasets.dataset_factory._get_dataset_factory",
        lambda cfg: FakeFactory(),
    )

    result = create_pytorch_datasets_from_config(config)

    assert result is expected_bundle
