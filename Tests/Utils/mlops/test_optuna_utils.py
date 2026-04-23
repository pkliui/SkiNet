import pytest
import torch
from unittest.mock import MagicMock
from SkiNet.Utils.mlops.optuna_utils import _collect_trainer_metrics, scale_lr


# ── scale_lr ────────────────────────────────────────────────────────────────

class TestScaleLr:
    def test_identity_at_base_batch_size(self) -> None:
        assert scale_lr(3e-4, batch_size=16, base_batch_size=16) == pytest.approx(3e-4)

    def test_scales_up_with_larger_batch(self) -> None:
        assert scale_lr(3e-4, batch_size=32, base_batch_size=16) == pytest.approx(6e-4)

    def test_scales_down_with_smaller_batch(self) -> None:
        assert scale_lr(3e-4, batch_size=8, base_batch_size=16) == pytest.approx(1.5e-4)

    def test_default_base_batch_size_is_16(self) -> None:
        assert scale_lr(1e-3, batch_size=16) == pytest.approx(1e-3)

    def test_zero_lr_returns_zero(self) -> None:
        assert scale_lr(0.0, batch_size=32) == pytest.approx(0.0)

    def test_zero_base_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="base_batch_size must be positive"):
            scale_lr(1e-3, batch_size=16, base_batch_size=0)

    def test_negative_base_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="base_batch_size must be positive"):
            scale_lr(1e-3, batch_size=16, base_batch_size=-1)


# ── _collect_trainer_metrics ─────────────────────────────────────────────────

def make_trainer(callback: dict = {}, logged: dict = {}, progress_bar: dict = {}) -> MagicMock:
    trainer = MagicMock()
    trainer.callback_metrics = callback
    trainer.logged_metrics = logged
    trainer.progress_bar_metrics = progress_bar
    return trainer


class TestCollectTrainerMetrics:
    def test_collects_scalar_tensor(self) -> None:
        trainer = make_trainer(callback={"val_loss": torch.tensor(0.42)})
        result = _collect_trainer_metrics(trainer)
        assert result == {"val_loss": pytest.approx(0.42)}

    def test_collects_float_value(self) -> None:
        trainer = make_trainer(logged={"val_dice": 0.87})
        result = _collect_trainer_metrics(trainer)
        assert result == {"val_dice": pytest.approx(0.87)}

    def test_collects_int_value(self) -> None:
        trainer = make_trainer(progress_bar={"epoch": 5})
        result = _collect_trainer_metrics(trainer)
        assert result == {"epoch": pytest.approx(5.0)}

    def test_skips_multielement_tensor(self) -> None:
        trainer = make_trainer(callback={"bad_metric": torch.tensor([0.1, 0.2])})
        result = _collect_trainer_metrics(trainer)
        assert "bad_metric" not in result

    def test_merges_all_three_sources(self) -> None:
        trainer = make_trainer(
            callback={"val_loss": torch.tensor(0.5)},
            logged={"val_dice": 0.9},
            progress_bar={"epoch": 3},
        )
        result = _collect_trainer_metrics(trainer)
        assert set(result.keys()) == {"val_loss", "val_dice", "epoch"}

    def test_conflicting_duplicate_key_raises(self) -> None:
        trainer = make_trainer(
            callback={"val_loss": torch.tensor(0.5)},
            progress_bar={"val_loss": 0.9},
        )
        with pytest.raises(ValueError, match="Conflicting values for metric 'val_loss'"):
            _collect_trainer_metrics(trainer)

    def test_identical_duplicate_key_does_not_raise(self) -> None:
        trainer = make_trainer(
            callback={"val_loss": torch.tensor(0.9)},
            progress_bar={"val_loss": 0.9},
        )
        result = _collect_trainer_metrics(trainer)
        assert result["val_loss"] == pytest.approx(0.9)

    def test_empty_sources_returns_empty(self) -> None:
        trainer = make_trainer()
        assert _collect_trainer_metrics(trainer) == {}

    def test_tensor_is_detached_and_on_cpu(self) -> None:
        # ensures no grad/device issues when converting
        t = torch.tensor(1.23, requires_grad=True)
        trainer = make_trainer(callback={"metric": t})
        result = _collect_trainer_metrics(trainer)
        assert result["metric"] == pytest.approx(1.23)
