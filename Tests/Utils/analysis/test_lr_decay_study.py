"""Tests for SkiNet.Utils.analysis.lr_decay_study."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from SkiNet.Utils.analysis.lr_decay_study import (
    DecayCondition,
    _epoch_series,
    build_decay_comparison,
    load_decay_conditions,
    plot_decay_dynamics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RUN_UUID = "aabbccdd-0000-0000-0000-000000000001"

MONITOR = "val_dice"


def _metrics_df(run_uuid: str = RUN_UUID, n_epochs: int = 5) -> pd.DataFrame:
    rows = []
    for ep in range(n_epochs):
        for key, val in [
            (MONITOR, 0.70 + ep * 0.01),
            ("lr-Adam", 3e-4),
            ("train_dice", 0.75 + ep * 0.01),
        ]:
            rows.append({"run_uuid": run_uuid, "key": key, "value": val, "step": ep, "timestamp": ep})
    return pd.DataFrame(rows)


def _summary_row(run_uuid: str = RUN_UUID, tail_mean: float = 0.75, tail_std: float = 0.01) -> pd.Series:
    return pd.Series({
        "run_uuid": run_uuid,
        "encoder": "classical",
        "merge": "attention_gate",
        "val_dice_tail_mean": tail_mean,
        "val_dice_tail_std": tail_std,
        "val_dice_max": 0.80,
        "val_dice_max_epoch": 5,
        "generalization_gap_final": 0.05,
        "samples_per_sec": 130.0,
        "duration_min": 45.0,
    })


def _params_df(run_uuid: str = RUN_UUID) -> pd.DataFrame:
    return pd.DataFrame([
        {"run_uuid": run_uuid, "key": "lr", "value": "0.0003"},
        {"run_uuid": run_uuid, "key": "scheduler", "value": "constant"},
    ])


def _make_condition(
    key: str = "constant",
    label: str = "Constant LR",
    colour: str = "#1f77b4",
    tail_mean: float = 0.75,
    n_epochs: int = 20,
) -> DecayCondition:
    row = _summary_row(tail_mean=tail_mean)
    params = _params_df().set_index("key")["value"]
    return DecayCondition(
        key=key,
        label=label,
        colour=colour,
        summary=row,
        monitor=pd.DataFrame({"epoch": np.arange(1, n_epochs + 1), "value": np.linspace(0.70, 0.80, n_epochs)}),
        lr=pd.DataFrame({"epoch": np.arange(1, n_epochs + 1), "value": np.full(n_epochs, 3e-4)}),
        train_dice=pd.DataFrame({"epoch": np.arange(1, n_epochs + 1), "value": np.linspace(0.75, 0.85, n_epochs)}),
        params=params,
    )


# ---------------------------------------------------------------------------
# _epoch_series
# ---------------------------------------------------------------------------

class TestEpochSeries:
    def test_returns_epoch_value_columns(self) -> None:
        metrics = _metrics_df(n_epochs=3)
        result = _epoch_series(metrics, RUN_UUID, MONITOR)
        assert list(result.columns) == ["epoch", "value"]

    def test_epoch_numbering_starts_at_1(self) -> None:
        metrics = _metrics_df(n_epochs=4)
        result = _epoch_series(metrics, RUN_UUID, MONITOR)
        assert result["epoch"].tolist() == [1, 2, 3, 4]

    def test_values_ordered_by_step(self) -> None:
        metrics = _metrics_df(n_epochs=5)
        result = _epoch_series(metrics, RUN_UUID, MONITOR)
        assert result["value"].tolist() == sorted(result["value"].tolist())

    def test_unknown_key_returns_empty(self) -> None:
        metrics = _metrics_df(n_epochs=3)
        result = _epoch_series(metrics, RUN_UUID, "nonexistent_key")
        assert len(result) == 0

    def test_unknown_run_returns_empty(self) -> None:
        metrics = _metrics_df(n_epochs=3)
        result = _epoch_series(metrics, "bad-uuid", MONITOR)
        assert len(result) == 0

    def test_length_matches_logged_steps(self) -> None:
        metrics = _metrics_df(n_epochs=7)
        result = _epoch_series(metrics, RUN_UUID, MONITOR)
        assert len(result) == 7


# ---------------------------------------------------------------------------
# load_decay_conditions
# ---------------------------------------------------------------------------

def _make_tables(run_uuid: str = RUN_UUID, n_epochs: int = 10) -> dict[str, pd.DataFrame]:
    return {
        "runs": pd.DataFrame({
            "run_uuid": [run_uuid],
            "run_name": ["run_seed100"],
            "lifecycle_stage": ["active"],
        }),
        "metrics": _metrics_df(run_uuid=run_uuid, n_epochs=n_epochs),
        "params": _params_df(run_uuid=run_uuid),
        "latest": pd.DataFrame(columns=["run_uuid", "key", "value"]),
    }


class TestLoadDecayConditions:
    CONDITIONS = {
        "constant": {"label": "Constant LR", "colour": "#1f77b4", "db": Path("/fake/constant.db")},
    }

    @patch("SkiNet.Utils.analysis.lr_decay_study.summarize_runs")
    @patch("SkiNet.Utils.analysis.lr_decay_study.load_mlflow_tables")
    def test_returns_dict_keyed_by_condition(self, mock_load: Any, mock_summarize: Any) -> None:
        mock_load.return_value = _make_tables()
        mock_summarize.return_value = pd.DataFrame([_summary_row()])
        result = load_decay_conditions(self.CONDITIONS, monitor=MONITOR, seed=100)
        assert "constant" in result

    @patch("SkiNet.Utils.analysis.lr_decay_study.summarize_runs")
    @patch("SkiNet.Utils.analysis.lr_decay_study.load_mlflow_tables")
    def test_condition_key_label_colour_set(self, mock_load: Any, mock_summarize: Any) -> None:
        mock_load.return_value = _make_tables()
        mock_summarize.return_value = pd.DataFrame([_summary_row()])
        result = load_decay_conditions(self.CONDITIONS, monitor=MONITOR, seed=100)
        cond = result["constant"]
        assert cond.key == "constant"
        assert cond.label == "Constant LR"
        assert cond.colour == "#1f77b4"

    @patch("SkiNet.Utils.analysis.lr_decay_study.summarize_runs")
    @patch("SkiNet.Utils.analysis.lr_decay_study.load_mlflow_tables")
    def test_monitor_series_attached(self, mock_load: Any, mock_summarize: Any) -> None:
        mock_load.return_value = _make_tables(n_epochs=5)
        mock_summarize.return_value = pd.DataFrame([_summary_row()])
        result = load_decay_conditions(self.CONDITIONS, monitor=MONITOR, seed=100)
        assert list(result["constant"].monitor.columns) == ["epoch", "value"]
        assert len(result["constant"].monitor) == 5

    @patch("SkiNet.Utils.analysis.lr_decay_study.summarize_runs")
    @patch("SkiNet.Utils.analysis.lr_decay_study.load_mlflow_tables")
    def test_lr_series_attached(self, mock_load: Any, mock_summarize: Any) -> None:
        mock_load.return_value = _make_tables(n_epochs=5)
        mock_summarize.return_value = pd.DataFrame([_summary_row()])
        result = load_decay_conditions(self.CONDITIONS, monitor=MONITOR, seed=100)
        assert list(result["constant"].lr.columns) == ["epoch", "value"]
        assert len(result["constant"].lr) == 5

    @patch("SkiNet.Utils.analysis.lr_decay_study.summarize_runs")
    @patch("SkiNet.Utils.analysis.lr_decay_study.load_mlflow_tables")
    def test_seed_filter_selects_correct_run(self, mock_load: Any, mock_summarize: Any) -> None:
        run_a, run_b = "uuid-a", "uuid-b"
        tables = {
            "runs": pd.DataFrame({
                "run_uuid": [run_a, run_b],
                "run_name": ["run_seed100", "run_seed200"],
                "lifecycle_stage": ["active", "active"],
            }),
            "metrics": pd.concat([_metrics_df(run_a, 5), _metrics_df(run_b, 5)], ignore_index=True),
            "params": pd.concat([_params_df(run_a), _params_df(run_b)], ignore_index=True),
            "latest": pd.DataFrame(columns=["run_uuid", "key", "value"]),
        }
        summary_df = pd.DataFrame([_summary_row(run_a), _summary_row(run_b)])
        mock_load.return_value = tables
        mock_summarize.return_value = summary_df
        result = load_decay_conditions(self.CONDITIONS, monitor=MONITOR, seed=100)
        assert result["constant"].summary["run_uuid"] == run_a

    @patch("SkiNet.Utils.analysis.lr_decay_study.summarize_runs")
    @patch("SkiNet.Utils.analysis.lr_decay_study.load_mlflow_tables")
    def test_merge_filter_applied_when_spec_has_merge(self, mock_load: Any, mock_summarize: Any) -> None:
        run_a, run_b = "uuid-ag", "uuid-add"
        tables = {
            "runs": pd.DataFrame({
                "run_uuid": [run_a, run_b],
                "run_name": ["run_seed100", "run_seed100"],
                "lifecycle_stage": ["active", "active"],
            }),
            "metrics": pd.concat([_metrics_df(run_a, 5), _metrics_df(run_b, 5)], ignore_index=True),
            "params": pd.concat([_params_df(run_a), _params_df(run_b)], ignore_index=True),
            "latest": pd.DataFrame(columns=["run_uuid", "key", "value"]),
        }
        summary_df = pd.DataFrame([
            {**_summary_row(run_a).to_dict(), "merge": "attention_gate"},
            {**_summary_row(run_b).to_dict(), "merge": "add"},
        ])
        conditions_with_merge = {
            "constant": {
                "label": "Constant LR", "colour": "#1f77b4",
                "db": Path("/fake/constant.db"), "merge": "attention_gate",
            },
        }
        mock_load.return_value = tables
        mock_summarize.return_value = summary_df
        result = load_decay_conditions(conditions_with_merge, monitor=MONITOR, seed=100)
        assert result["constant"].summary["run_uuid"] == run_a


# ---------------------------------------------------------------------------
# build_decay_comparison
# ---------------------------------------------------------------------------

class TestBuildDecayComparison:
    def _conditions(self) -> dict[str, DecayCondition]:
        return {
            "constant": _make_condition("constant", "Constant LR", "#1f77b4", tail_mean=0.750),
            "rop": _make_condition("rop", "ReduceLROnPlateau", "#ff7f0e", tail_mean=0.758),
            "cosine": _make_condition("cosine", "CosineAnnealingLR", "#2ca02c", tail_mean=0.762),
        }

    def test_one_row_per_condition(self) -> None:
        result = build_decay_comparison(self._conditions(), MONITOR, "constant")
        assert len(result) == 3

    def test_index_is_condition_labels(self) -> None:
        result = build_decay_comparison(self._conditions(), MONITOR, "constant")
        assert set(result.index) == {"Constant LR", "ReduceLROnPlateau", "CosineAnnealingLR"}

    def test_baseline_delta_is_zero(self) -> None:
        result = build_decay_comparison(self._conditions(), MONITOR, "constant")
        assert result.loc["Constant LR", "Δplateau_vs_base"] == pytest.approx(0.0)

    def test_delta_sign_correct(self) -> None:
        result = build_decay_comparison(self._conditions(), MONITOR, "constant")
        # cosine tail_mean=0.762 > constant tail_mean=0.750
        assert result.loc["CosineAnnealingLR", "Δplateau_vs_base"] == pytest.approx(0.012)

    def test_plateau_dice_column_present(self) -> None:
        result = build_decay_comparison(self._conditions(), MONITOR, "constant")
        assert "plateau_dice" in result.columns

    def test_sps_delta_column_present(self) -> None:
        result = build_decay_comparison(self._conditions(), MONITOR, "constant")
        assert "Δsps_vs_base" in result.columns


# ---------------------------------------------------------------------------
# plot_decay_dynamics
# ---------------------------------------------------------------------------

class TestPlotDecayDynamics:
    def _conditions(self, n_epochs: int = 20) -> dict[str, DecayCondition]:
        return {
            "constant": _make_condition("constant", "Constant LR", "#1f77b4", n_epochs=n_epochs),
            "cosine": _make_condition("cosine", "CosineAnnealingLR", "#2ca02c", n_epochs=n_epochs),
        }

    def test_returns_matplotlib_figure(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        fig = plot_decay_dynamics(self._conditions(), MONITOR, "cosine", title="Test")
        assert isinstance(fig, Figure)

    def test_two_axes(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_decay_dynamics(self._conditions(), MONITOR, "cosine", title="Test")
        assert len(fig.axes) == 2

    def test_saves_to_disk(self, tmp_path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        out = tmp_path / "fig.png"
        plot_decay_dynamics(self._conditions(), MONITOR, "cosine", title="Test", save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_no_save_when_path_is_none(self, tmp_path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        plot_decay_dynamics(self._conditions(), MONITOR, "cosine", title="Test", save_path=None)
        assert not any(tmp_path.iterdir())
