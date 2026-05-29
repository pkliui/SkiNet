"""Tests for SkiNet.Utils.analysis.plotting."""

import pandas as pd
import pytest
from matplotlib.figure import Figure

from SkiNet.Utils.analysis.plotting import (
    plot_accuracy_throughput,
    plot_architecture_heatmap,
    plot_generalization_gap,
    plot_learning_curves,
    plot_train_val_overlay,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def run_summary() -> pd.DataFrame:
    return pd.DataFrame({
        "encoder": ["none", "none", "full", "full"],
        "merge": ["none", "full", "none", "full"],
        "best_val_dice": [0.80, 0.82, 0.78, 0.85],
        "best_val_best_dice_at_threshold": [0.79, 0.81, 0.77, 0.84],
        "generalization_gap_final": [0.05, 0.03, 0.07, 0.02],
        "samples_per_sec": [12.0, 11.5, 13.0, 10.8],
    })


@pytest.fixture()
def history() -> pd.DataFrame:
    rows = []
    for arch in ["none+none", "full+full"]:
        for epoch in range(1, 4):
            rows.append({"architecture": arch, "key": "val_dice", "epoch": epoch, "value": 0.7 + epoch * 0.01})
            rows.append({"architecture": arch, "key": "train_dice", "epoch": epoch, "value": 0.75 + epoch * 0.01})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# plot_architecture_heatmap
# ---------------------------------------------------------------------------

class TestPlotArchitectureHeatmap:
    def test_returns_figure(self, run_summary: pd.DataFrame) -> None:
        fig = plot_architecture_heatmap(run_summary, "best_val_dice", "Test heatmap")
        assert isinstance(fig, Figure)

    def test_axes_labels(self, run_summary: pd.DataFrame) -> None:
        fig = plot_architecture_heatmap(run_summary, "best_val_dice", "My title")
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Merge residual mode"
        assert ax.get_ylabel() == "Encoder residual mode"
        assert ax.get_title() == "My title"

    def test_cell_annotation_format(self, run_summary: pd.DataFrame) -> None:
        fig = plot_architecture_heatmap(run_summary, "best_val_dice", "t")
        ax = fig.axes[0]
        for text in ax.texts:
            val = text.get_text()
            assert val == "n/a" or len(val.split(".")[1]) == 4

    def test_nan_cell_annotated_as_na(self) -> None:
        df = pd.DataFrame({
            "encoder": ["none", "none"],
            "merge": ["none", "full"],
            "metric": [0.80, float("nan")],
        })
        fig = plot_architecture_heatmap(df, "metric", "t")
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert "n/a" in texts

    def test_duplicate_encoder_merge_raises(self) -> None:
        df = pd.DataFrame({
            "encoder": ["none", "none"],
            "merge": ["none", "none"],
            "best_val_dice": [0.80, 0.82],
        })
        with pytest.raises(ValueError):
            plot_architecture_heatmap(df, "best_val_dice", "t")


# ---------------------------------------------------------------------------
# plot_learning_curves
# ---------------------------------------------------------------------------

class TestPlotLearningCurves:
    def test_returns_figure(self, history: pd.DataFrame) -> None:
        fig = plot_learning_curves(history, "val_dice", "Dice", "Val Dice")
        assert isinstance(fig, Figure)

    def test_axes_labels(self, history: pd.DataFrame) -> None:
        fig = plot_learning_curves(history, "val_dice", "Dice score", "Learning curves")
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Epoch"
        assert ax.get_ylabel() == "Dice score"
        assert ax.get_title() == "Learning curves"

    def test_unknown_key_produces_empty_axes(self, history: pd.DataFrame) -> None:
        fig = plot_learning_curves(history, "nonexistent_key", "Y", "t")
        assert len(fig.axes[0].lines) == 0


# ---------------------------------------------------------------------------
# plot_train_val_overlay
# ---------------------------------------------------------------------------

class TestPlotTrainValOverlay:
    def test_returns_figure(self, history: pd.DataFrame) -> None:
        fig = plot_train_val_overlay(history)
        assert isinstance(fig, Figure)

    def test_default_title(self, history: pd.DataFrame) -> None:
        fig = plot_train_val_overlay(history)
        assert fig.axes[0].get_title() == "Train vs validation Dice by epoch"

    def test_custom_title(self, history: pd.DataFrame) -> None:
        fig = plot_train_val_overlay(history, title="Custom")
        assert fig.axes[0].get_title() == "Custom"

    def test_axes_labels(self, history: pd.DataFrame) -> None:
        fig = plot_train_val_overlay(history)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Epoch"
        assert ax.get_ylabel() == "Dice"

    def test_missing_train_key_does_not_crash(self) -> None:
        df = pd.DataFrame({
            "architecture": ["A", "A"],
            "key": ["val_dice", "val_dice"],
            "epoch": [1, 2],
            "value": [0.7, 0.75],
        })
        fig = plot_train_val_overlay(df)
        assert isinstance(fig, Figure)

    def test_empty_history_does_not_crash(self) -> None:
        df = pd.DataFrame(columns=["architecture", "key", "epoch", "value"])
        fig = plot_train_val_overlay(df)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_generalization_gap
# ---------------------------------------------------------------------------

class TestPlotGeneralizationGap:
    def test_returns_figure(self, run_summary: pd.DataFrame) -> None:
        fig = plot_generalization_gap(run_summary)
        assert isinstance(fig, Figure)

    def test_axes_labels(self, run_summary: pd.DataFrame) -> None:
        fig = plot_generalization_gap(run_summary)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "train Dice - val Dice"
        assert ax.get_title() == "Final train-validation Dice gap"

    def test_bars_sorted_by_best_val(self, run_summary: pd.DataFrame) -> None:
        import matplotlib.patches as mpatches
        fig = plot_generalization_gap(run_summary)
        ax = fig.axes[0]
        bar_widths = [p.get_width() for p in ax.patches if isinstance(p, mpatches.Rectangle)]
        expected: list[float] = list(run_summary.sort_values("best_val_best_dice_at_threshold")["generalization_gap_final"])
        assert bar_widths == pytest.approx(expected)

    def test_value_annotations_match_gap_values(self, run_summary: pd.DataFrame) -> None:
        fig = plot_generalization_gap(run_summary)
        ax = fig.axes[0]
        annotated = {float(t.get_text()) for t in ax.texts}
        expected = {round(v, 3) for v in run_summary["generalization_gap_final"]}
        assert annotated == expected


# ---------------------------------------------------------------------------
# plot_accuracy_throughput
# ---------------------------------------------------------------------------

class TestPlotAccuracyThroughput:
    @pytest.fixture()
    def summary_with_tail(self) -> pd.DataFrame:
        return pd.DataFrame({
            "encoder": ["none", "none", "full", "full"],
            "merge": ["none", "full", "none", "full"],
            "samples_per_sec": [12.0, 11.5, 13.0, 10.8],
            "best_val_best_dice_at_threshold": [0.79, 0.81, 0.77, 0.84],
            "best_val_best_dice_at_threshold_tail_mean": [0.78, 0.80, 0.76, 0.83],
            "best_val_best_dice_at_threshold_tail_std": [0.01, 0.01, 0.02, 0.01],
        })

    def test_returns_figure(self, run_summary: pd.DataFrame) -> None:
        fig = plot_accuracy_throughput(run_summary)
        assert isinstance(fig, Figure)

    def test_axes_labels(self, run_summary: pd.DataFrame) -> None:
        fig = plot_accuracy_throughput(run_summary)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Final training throughput, samples/sec"
        assert ax.get_ylabel() == "Validation Dice-at-threshold"
        assert ax.get_title() == "Accuracy-throughput frontier"

    def test_custom_monitor_column(self) -> None:
        df = pd.DataFrame({
            "encoder": ["none", "full"],
            "merge": ["none", "none"],
            "samples_per_sec": [12.0, 13.0],
            "best_val_dice": [0.80, 0.78],
        })
        fig = plot_accuracy_throughput(df, monitor="val_dice")
        assert isinstance(fig, Figure)

    def test_no_tail_columns_no_error_bars(self, run_summary: pd.DataFrame) -> None:
        from matplotlib.container import ErrorbarContainer
        fig = plot_accuracy_throughput(run_summary)
        assert not any(isinstance(c, ErrorbarContainer) for c in fig.axes[0].containers)

    def test_tail_columns_produce_error_bars(self, summary_with_tail: pd.DataFrame) -> None:
        from matplotlib.container import ErrorbarContainer
        fig = plot_accuracy_throughput(summary_with_tail)
        assert any(isinstance(c, ErrorbarContainer) for c in fig.axes[0].containers)
