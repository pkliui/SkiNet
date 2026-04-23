from types import SimpleNamespace
from typing import Any
import pytest
from unittest.mock import MagicMock

from SkiNet.ML.configs.train_configs.train_config import TrainConfig
from SkiNet.ML.dataloaders.create_dataloaders import create_dataloaders_from_datasets
from SkiNet.ML.datasets.dataset_factory import DatasetSplit


def _make_dataset_split() -> DatasetSplit:
    """Minimal fake DatasetSplit with dummy train/val/test datasets."""
    return DatasetSplit(
        train=MagicMock(name="train_ds"),
        val=MagicMock(name="val_ds"),
        test=MagicMock(name="test_ds"),
        splits=MagicMock(),
    )


def _make_train_cfg(**overrides: Any) -> TrainConfig:
    return TrainConfig(**overrides)


# --- create_dataloaders_from_datasets ---

def test_create_dataloaders_returns_all_three_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    """DataLoaders container has train/val/test populated."""
    datasets = _make_dataset_split()
    cfg = _make_train_cfg()

    # monkeypatch where RepeatDataLoader is used not wehre it is originating from!
    monkeypatch.setattr(
        "SkiNet.ML.dataloaders.create_dataloaders.RepeatDataLoader",
        lambda *a, **kw: SimpleNamespace(**kw),
    )

    result = create_dataloaders_from_datasets(datasets, cfg)

    assert result.train is not None
    assert result.val is not None
    assert result.test is not None


def test_create_dataloaders_passes_correct_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each loader receives the correct dataset."""
    datasets = _make_dataset_split()
    cfg = _make_train_cfg()
    created: list[Any] = []

    class FakeRepeatDataLoader:
        def __init__(self, dataset: Any, **kw: Any) -> None:
            created.append(dataset)

    monkeypatch.setattr(
        "SkiNet.ML.dataloaders.create_dataloaders.RepeatDataLoader", FakeRepeatDataLoader
    )

    create_dataloaders_from_datasets(datasets, cfg)

    assert created[0] is datasets.train
    assert created[1] is datasets.val
    assert created[2] is datasets.test


def test_create_dataloaders_passes_cfg_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """batch_size, num_workers, pin_memory are forwarded from TrainConfig."""
    datasets = _make_dataset_split()
    cfg = _make_train_cfg(batch_size=4, num_workers=0, pin_memory=False)
    created: list[dict[str, Any]] = []

    class FakeRepeatDataLoader:
        def __init__(self, dataset: Any, **kw: Any) -> None:
            created.append(kw)

    monkeypatch.setattr(
        "SkiNet.ML.dataloaders.create_dataloaders.RepeatDataLoader", FakeRepeatDataLoader
    )

    create_dataloaders_from_datasets(datasets, cfg)

    for kw in created:
        assert kw["batch_size"] == 4
        assert kw["num_workers"] == 0
        assert kw["pin_memory"] is False


def test_create_dataloaders_train_shuffles_val_test_do_not(monkeypatch: pytest.MonkeyPatch) -> None:
    """Train loader has shuffle=True; val and test have shuffle=False."""
    datasets = _make_dataset_split()
    cfg = _make_train_cfg()
    created: list[dict[str, Any]] = []

    class FakeRepeatDataLoader:
        def __init__(self, dataset: Any, **kw: Any) -> None:
            created.append(kw)

    monkeypatch.setattr(
        "SkiNet.ML.dataloaders.create_dataloaders.RepeatDataLoader", FakeRepeatDataLoader
    )

    create_dataloaders_from_datasets(datasets, cfg)

    assert created[0]["shuffle"] is True   # train
    assert created[1]["shuffle"] is False  # val
    assert created[2]["shuffle"] is False  # test


@pytest.mark.parametrize(("num_workers", "prefetch_factor_in", "expected_prefetch"), [
    (0, None, None),  # already None, validator keeps it None
    (0, 4, None),  # validator forces to None when num_workers=0
    (2, 4, 4),  # forwarded as-is when workers > 0
    (2, None, None),  # user explicitly set None, respected
])
def test_create_dataloaders_prefetch_factor_conditioned_on_num_workers(
    monkeypatch: pytest.MonkeyPatch,
    num_workers: int,
    prefetch_factor_in: int | None,
    expected_prefetch: int | None,
) -> None:
    """prefetch_factor is forced to None when num_workers=0, otherwise forwarded unchanged."""
    datasets = _make_dataset_split()
    cfg = _make_train_cfg(num_workers=num_workers, prefetch_factor=prefetch_factor_in)
    created: list[dict[str, Any]] = []

    class FakeRepeatDataLoader:
        def __init__(self, dataset: Any, **kw: Any) -> None:
            created.append(kw)

    monkeypatch.setattr(
        "SkiNet.ML.dataloaders.create_dataloaders.RepeatDataLoader", FakeRepeatDataLoader
    )

    create_dataloaders_from_datasets(datasets, cfg)

    assert len(created) > 0, "No dataloaders were created"
    for kw in created:
        assert kw.get("prefetch_factor") == expected_prefetch
