"""Tests for SkiNet.Utils.analysis.plotting."""

from typing import Any
from SkiNet.Utils.analysis.plotting import set_paper_style
from SkiNet.Utils.analysis.plotting import plot_paired_forest, plot_paired_slopegraph
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from SkiNet.Utils.analysis.plotting import (
    plot_accuracy_throughput,
    plot_architecture_heatmap,
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
        "val_dice_max": [0.80, 0.82, 0.78, 0.85],
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
        fig = plot_architecture_heatmap(run_summary, "val_dice_max", "Test heatmap")
        assert isinstance(fig, Figure)

    def test_axes_labels(self, run_summary: pd.DataFrame) -> None:
        fig = plot_architecture_heatmap(run_summary, "val_dice_max", "My title")
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Merge residual mode"
        assert ax.get_ylabel() == "Encoder residual mode"
        assert ax.get_title() == "My title"

    def test_cell_annotation_format(self, run_summary: pd.DataFrame) -> None:
        fig = plot_architecture_heatmap(run_summary, "val_dice_max", "t")
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
            "val_dice_max": [0.80, 0.82],
        })
        with pytest.raises(ValueError):
            plot_architecture_heatmap(df, "val_dice_max", "t")


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
        fig = plot_train_val_overlay(history, monitor="val_dice")
        assert isinstance(fig, Figure)

    def test_default_title(self, history: pd.DataFrame) -> None:
        fig = plot_train_val_overlay(history, monitor="val_dice")
        assert fig.axes[0].get_title() == "Train vs validation Dice by epoch"

    def test_custom_title(self, history: pd.DataFrame) -> None:
        fig = plot_train_val_overlay(history, monitor="val_dice", title="Custom")
        assert fig.axes[0].get_title() == "Custom"

    def test_axes_labels(self, history: pd.DataFrame) -> None:
        fig = plot_train_val_overlay(history, monitor="val_dice")
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
        fig = plot_train_val_overlay(df, monitor="val_dice")
        assert isinstance(fig, Figure)

    def test_empty_history_does_not_crash(self) -> None:
        df = pd.DataFrame(columns=["architecture", "key", "epoch", "value"])
        fig = plot_train_val_overlay(df, monitor="val_dice")
        assert isinstance(fig, Figure)


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
            "val_dice_max": [0.79, 0.81, 0.77, 0.84],
            "val_dice_tail_mean": [0.78, 0.80, 0.76, 0.83],
            "val_dice_tail_std": [0.01, 0.01, 0.02, 0.01],
        })

    def test_returns_figure(self, run_summary: pd.DataFrame) -> None:
        fig = plot_accuracy_throughput(run_summary, monitor="val_dice")
        assert isinstance(fig, Figure)

    def test_axes_labels(self, run_summary: pd.DataFrame) -> None:
        fig = plot_accuracy_throughput(run_summary, monitor="val_dice")
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Final training throughput, samples/sec"
        assert ax.get_ylabel() == "Validation Dice-at-threshold"
        assert ax.get_title() == "Accuracy-throughput frontier  [colour = RdYlGn by tail-mean rank]"

    def test_custom_monitor_column(self) -> None:
        df = pd.DataFrame({
            "encoder": ["none", "full"],
            "merge": ["none", "none"],
            "samples_per_sec": [12.0, 13.0],
            "val_dice_max": [0.80, 0.78],
        })
        fig = plot_accuracy_throughput(df, monitor="val_dice")
        assert isinstance(fig, Figure)

    def test_no_tail_columns_no_error_bars(self, run_summary: pd.DataFrame) -> None:
        from matplotlib.container import ErrorbarContainer
        fig = plot_accuracy_throughput(run_summary, monitor="val_dice")
        assert not any(isinstance(c, ErrorbarContainer) for c in fig.axes[0].containers)

    def test_tail_columns_produce_error_bars(self, summary_with_tail: pd.DataFrame) -> None:
        from matplotlib.container import ErrorbarContainer
        fig = plot_accuracy_throughput(summary_with_tail, monitor="val_dice")
        assert any(isinstance(c, ErrorbarContainer) for c in fig.axes[0].containers)


# ---------------------------------------------------------------------------
# Shared fixtures for paired plots
# ---------------------------------------------------------------------------


_PALETTE = {"A": "#d1495b", "B": "#30638e"}
_SEEDS = [0, 1, 2, 3]
_D = np.array([0.010, -0.005, 0.008, 0.003])


@pytest.fixture()
def paired_runs() -> pd.DataFrame:
    rows = []
    for i, seed in enumerate(_SEEDS):
        rows.append({"seed": seed, "arch": "A",
                     "val_dice": 0.80 + _D[i] / 2, "val_iou": 0.70 + _D[i] / 2})
        rows.append({"seed": seed, "arch": "B",
                     "val_dice": 0.80 - _D[i] / 2, "val_iou": 0.70 - _D[i] / 2})
    return pd.DataFrame(rows)


@pytest.fixture()
def paired_results() -> pd.DataFrame:
    return pd.DataFrame({
        "delta_a_minus_b": [0.005, 0.003, -0.002],
        "boot_lo": [0.001, 0.001, -0.006],
        "boot_hi": [0.009, 0.007, 0.002],
        "wilcoxon_p": [0.031, 0.250, 0.500],
        "cohen_dz": [0.80, 0.30, -0.20],
    }, index=pd.Index(["val_dice", "val_iou", "gen_gap"], name="metric"))


# ---------------------------------------------------------------------------
# plot_paired_slopegraph
# ---------------------------------------------------------------------------

class TestPlotPairedSlopegraph:
    def test_returns_figure(self, paired_runs: pd.DataFrame) -> None:
        fig = plot_paired_slopegraph(
            paired_runs,
            [("val_dice", "Dice"), ("val_iou", "IoU")],
            arch_a="A", arch_b="B", seeds=_SEEDS, palette=_PALETTE,
        )
        assert isinstance(fig, Figure)

    def test_one_panel_per_metric(self, paired_runs: pd.DataFrame) -> None:
        fig = plot_paired_slopegraph(
            paired_runs, [("val_dice", "Dice"), ("val_iou", "IoU")],
            arch_a="A", arch_b="B", seeds=_SEEDS, palette=_PALETTE,
        )
        assert len(fig.axes) == 2

    def test_single_metric_returns_single_panel(self, paired_runs: pd.DataFrame) -> None:
        fig = plot_paired_slopegraph(
            paired_runs, [("val_dice", "Dice")],
            arch_a="A", arch_b="B", seeds=_SEEDS, palette=_PALETTE,
        )
        assert len(fig.axes) == 1

    def test_title_contains_seed_count(self, paired_runs: pd.DataFrame) -> None:
        fig = plot_paired_slopegraph(
            paired_runs, [("val_dice", "Dice")],
            arch_a="A", arch_b="B", seeds=_SEEDS, palette=_PALETTE,
        )
        assert str(len(_SEEDS)) in fig.texts[0].get_text()

    def test_custom_seed_col(self, paired_runs: pd.DataFrame) -> None:
        df = paired_runs.rename(columns={"seed": "run_seed"})
        fig = plot_paired_slopegraph(
            df, [("val_dice", "Dice")],
            arch_a="A", arch_b="B", seeds=_SEEDS, palette=_PALETTE,
            seed_col="run_seed",
        )
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_paired_forest
# ---------------------------------------------------------------------------

class TestPlotPairedForest:
    _SPECS = [
        ("Dice", "val_dice", False),
        ("IoU", "val_iou", False),
        ("Gen-gap", "gen_gap", True),
    ]

    def test_returns_figure(self, paired_results: pd.DataFrame) -> None:
        fig = plot_paired_forest(
            paired_results, self._SPECS,
            arch_a="A", arch_b="B", n=4, palette=_PALETTE,
        )
        assert isinstance(fig, Figure)

    def test_correct_number_of_y_ticks(self, paired_results: pd.DataFrame) -> None:
        fig = plot_paired_forest(
            paired_results, self._SPECS,
            arch_a="A", arch_b="B", n=4, palette=_PALETTE,
        )
        ax = fig.axes[0]
        assert len(ax.get_yticks()) == len(self._SPECS)

    def test_y_labels_match_spec_display_names(self, paired_results: pd.DataFrame) -> None:
        fig = plot_paired_forest(
            paired_results, self._SPECS,
            arch_a="A", arch_b="B", n=4, palette=_PALETTE,
        )
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        for display_name, _, _ in self._SPECS:
            assert display_name in labels

    def test_custom_delta_col(self, paired_results: pd.DataFrame) -> None:
        df = paired_results.rename(columns={"delta_a_minus_b": "my_delta"})
        fig = plot_paired_forest(
            df, self._SPECS,
            arch_a="A", arch_b="B", n=4, palette=_PALETTE,
            delta_col="my_delta",
        )
        assert isinstance(fig, Figure)

    def test_flip_sign_negates_point(self, paired_results: pd.DataFrame) -> None:
        # gen_gap has flip=True; delta_a_minus_b=-0.002 → displayed as +0.002 → A better
        # Colour should be palette["A"] (positive direction)
        fig = plot_paired_forest(
            paired_results,
            [("Gen-gap", "gen_gap", True)],
            arch_a="A", arch_b="B", n=4, palette=_PALETTE,
        )
        # The annotation text should show a positive delta
        annot = [t.get_text() for t in fig.axes[0].texts if t.get_text().startswith("+")]
        assert len(annot) == 1


# ---------------------------------------------------------------------------
# Tufte Beautiful Evidence style — set_paper_style
# ---------------------------------------------------------------------------


class TestSetPaperStyle:
    def test_no_grid_after_call(self) -> None:
        set_paper_style()
        import matplotlib as mpl
        assert not mpl.rcParams["axes.grid"]

    def test_top_spine_off(self) -> None:
        set_paper_style()
        import matplotlib as mpl
        assert not mpl.rcParams["axes.spines.top"]

    def test_right_spine_off(self) -> None:
        set_paper_style()
        import matplotlib as mpl
        assert not mpl.rcParams["axes.spines.right"]

    def test_context_parameter_accepted(self) -> None:
        for ctx in ("paper", "notebook", "talk", "poster"):
            set_paper_style(context=ctx)  # must not raise

    def test_font_scale_accepted(self) -> None:
        set_paper_style(font_scale=1.5)  # must not raise


# ---------------------------------------------------------------------------
# Tufte slopegraph — no grid, no box, direct labels, range-frame ticks
# ---------------------------------------------------------------------------

class TestPlotPairedSlopegraphTufte:
    def _make(self, paired_runs: pd.DataFrame, **kw: Any) -> Figure:
        return plot_paired_slopegraph(
            paired_runs, [("val_dice", "Dice")],
            arch_a="A", arch_b="B", seeds=_SEEDS, palette=_PALETTE, **kw
        )

    def test_no_spine_visible(self, paired_runs: pd.DataFrame) -> None:
        fig = self._make(paired_runs)
        ax = fig.axes[0]
        visible = [s for s in ax.spines.values() if s.get_visible()]
        assert visible == [], "All four spines must be hidden (Tufte: erase chartjunk)"

    def test_no_x_ticks(self, paired_runs: pd.DataFrame) -> None:
        fig = self._make(paired_runs)
        ax = fig.axes[0]
        assert ax.get_xticks().size == 0, "x-ticks must be removed; architecture labels go below columns"

    def test_range_frame_y_ticks(self, paired_runs: pd.DataFrame) -> None:
        # Ticks must span only the observed data range, not the full ylim
        fig = self._make(paired_runs)
        ax = fig.axes[0]
        ticks = ax.get_yticks()
        assert len(ticks) == 4, "Exactly 4 range-frame y-ticks expected"
        # ticks should lie within the data range (allow small float tolerance)
        piv = paired_runs.pivot(index="seed", columns="arch", values="val_dice").loc[_SEEDS]
        y_lo = float(pd.concat([piv["A"], piv["B"]]).min())
        y_hi = float(pd.concat([piv["A"], piv["B"]]).max())
        assert ticks.min() >= y_lo - 1e-9
        assert ticks.max() <= y_hi + 1e-9

    def test_direct_architecture_labels_in_text(self, paired_runs: pd.DataFrame) -> None:
        # Architecture short names must appear as text objects on the axes
        fig = self._make(paired_runs)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert any("A" in t for t in texts), "label_a must appear as direct text"
        assert any("B" in t for t in texts), "label_b must appear as direct text"

    def test_mean_annotations_present(self, paired_runs: pd.DataFrame) -> None:
        # Mean value labels (4 decimal places) must be annotated on both sides
        fig = self._make(paired_runs)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        four_dp = [t for t in texts if "." in t and len(t.split(".")[-1]) == 4]
        assert len(four_dp) >= 2, "At least two 4-dp mean annotations expected (one per architecture)"

    def test_winner_lines_use_palette_colour(self, paired_runs: pd.DataFrame) -> None:
        # Lines where arch_a wins must be rendered in palette["A"] colour
        import matplotlib.colors as mc
        fig = self._make(paired_runs)
        ax = fig.axes[0]
        line_colours = {mc.to_hex(ln.get_color()) for ln in ax.lines}
        assert _PALETTE["A"].lower() in {c.lower() for c in line_colours}

    def test_save_path_writes_file(self, paired_runs: pd.DataFrame, tmp_path: Any) -> None:
        out = tmp_path / "slope.png"
        self._make(paired_runs, save_path=out)
        assert out.exists() and out.stat().st_size > 0

    def test_custom_title_overrides_default(self, paired_runs: pd.DataFrame) -> None:
        fig = self._make(paired_runs, title="My Custom Title")
        assert fig.texts[0].get_text() == "My Custom Title"


# ---------------------------------------------------------------------------
# Tufte forest plot — hairline CIs, no grid, marginal CI text, null line
# ---------------------------------------------------------------------------

class TestPlotPairedForestTufte:
    _SPECS = [
        ("Dice", "val_dice", False),
        ("IoU", "val_iou", False),
        ("Gen-gap", "gen_gap", True),
    ]

    def _make(self, paired_results: pd.DataFrame, specs: list[Any] | None = None, **kw: Any) -> Figure:
        return plot_paired_forest(
            paired_results, specs if specs is not None else self._SPECS,
            arch_a="A", arch_b="B", n=4, palette=_PALETTE, **kw
        )

    def test_no_grid(self, paired_results: pd.DataFrame) -> None:
        fig = self._make(paired_results)
        ax = fig.axes[0]
        assert not ax.xaxis.get_gridlines() or all(
            not ln.get_visible() for ln in ax.xaxis.get_gridlines()
        ), "No grid lines on x-axis (Tufte: null line alone orients the reader)"

    def test_left_spine_removed(self, paired_results: pd.DataFrame) -> None:
        fig = self._make(paired_results)
        ax = fig.axes[0]
        assert not ax.spines["left"].get_visible(), "Left spine must be removed (y labels are enough)"

    def test_null_line_is_thin(self, paired_results: pd.DataFrame) -> None:
        # axvline at x=0 must have lw < 1.2 — hairline, not bold
        fig = self._make(paired_results)
        ax = fig.axes[0]
        vlines = [ln for ln in ax.lines if list(np.asarray(ln.get_xdata())) == [0.0, 0.0]]
        assert vlines, "Null line (x=0) must be drawn"
        assert all(ln.get_linewidth() < 1.2 for ln in vlines), "Null line must be a hairline (lw < 1.2)"

    def test_ci_bounds_annotated_as_text(self, paired_results: pd.DataFrame) -> None:
        # Every row must produce a bracketed CI annotation like "[+0.001, +0.009]"
        fig = self._make(paired_results)
        ax = fig.axes[0]
        bracket_texts = [t.get_text() for t in ax.texts if t.get_text().startswith("[")]
        assert len(bracket_texts) == len(self._SPECS), (
            "Each metric must have a marginal CI annotation"
        )

    def test_ci_bars_are_hairlines(self, paired_results: pd.DataFrame) -> None:
        # All CI bar lines (excluding the null vline and scatter internals)
        # must have lw <= 1.0
        fig = self._make(paired_results)
        ax = fig.axes[0]
        # CI bars are horizontal lines at y = integer positions
        ci_lines = [
            ln for ln in ax.lines
            if len(np.asarray(ln.get_ydata())) == 2
            and np.asarray(ln.get_ydata())[0] == np.asarray(ln.get_ydata())[1]
        ]
        assert ci_lines, "CI bar lines must exist"
        assert all(ln.get_linewidth() <= 1.0 for ln in ci_lines), (
            "CI bars must be hairlines (lw ≤ 1.0) — Tufte: data-ink precision"
        )

    def test_direction_labels_in_annotations(self, paired_results: pd.DataFrame) -> None:
        # Architecture directional labels ("A better →" / "← B better") replace legend
        fig = self._make(paired_results)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert any("better" in t for t in texts), (
            "Direction labels ('better') must annotate the x-axis extremes"
        )

    def test_figure_height_scales_with_metrics(self, paired_results: pd.DataFrame) -> None:
        fig_3 = self._make(paired_results)
        fig_1 = self._make(paired_results, specs=[("Dice", "val_dice", False)])
        assert fig_3.get_figheight() > fig_1.get_figheight(), (
            "Figure height must grow with metric count (0.9 in/row)"
        )

    def test_save_path_writes_file(self, paired_results: pd.DataFrame, tmp_path: Any) -> None:
        out = tmp_path / "forest.png"
        self._make(paired_results, save_path=out)
        assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# row_stats integration — slopegraph stat stamp
# ---------------------------------------------------------------------------

class TestSlopegraphRowStats:
    _METRICS = [("val_dice", "Dice"), ("val_iou", "IoU")]

    def _make(self, paired_runs: pd.DataFrame, paired_results: pd.DataFrame, **kw: Any) -> Figure:
        row_stats = {col: paired_results.loc[col] for col, _ in self._METRICS
                     if col in paired_results.index}
        return plot_paired_slopegraph(
            paired_runs, self._METRICS,
            arch_a="A", arch_b="B", seeds=_SEEDS, palette=_PALETTE,
            row_stats=row_stats, alpha=0.05, **kw,
        )

    def test_stat_stamp_text_contains_p(
        self, paired_runs: pd.DataFrame, paired_results: pd.DataFrame
    ) -> None:
        fig = self._make(paired_runs, paired_results)
        all_text = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
        assert "p = " in all_text, "Stat stamp must include Wilcoxon p"

    def test_stat_stamp_text_contains_dz(
        self, paired_runs: pd.DataFrame, paired_results: pd.DataFrame
    ) -> None:
        fig = self._make(paired_runs, paired_results)
        all_text = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
        assert "d_z = " in all_text, "Stat stamp must include Cohen's d_z"

    def test_stat_stamp_text_contains_alpha(
        self, paired_runs: pd.DataFrame, paired_results: pd.DataFrame
    ) -> None:
        fig = self._make(paired_runs, paired_results)
        all_text = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
        assert "α = 0.05" in all_text, "Stat stamp must include α threshold"

    def test_no_stamp_without_row_stats(
        self, paired_runs: pd.DataFrame
    ) -> None:
        fig = plot_paired_slopegraph(
            paired_runs, self._METRICS,
            arch_a="A", arch_b="B", seeds=_SEEDS, palette=_PALETTE,
        )
        all_text = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
        assert "p = " not in all_text, "No stat stamp when row_stats is None"

    def test_significant_stamp_uses_winner_colour(
        self, paired_runs: pd.DataFrame, paired_results: pd.DataFrame
    ) -> None:
        # val_dice: p=0.031 < 0.05, delta > 0 → stamp should be col_a (#d1495b)
        import matplotlib.colors as mc
        fig = self._make(paired_runs, paired_results)
        stamp_colours = {
            mc.to_hex(t.get_color())
            for ax in fig.axes for t in ax.texts
            if "p = " in t.get_text()
        }
        winner_hex = _PALETTE["A"].lower()
        tie_hex = "#999999"
        assert any(c.lower() in (winner_hex, tie_hex) for c in stamp_colours)

    def test_tie_stamp_uses_grey(
        self, paired_runs: pd.DataFrame, paired_results: pd.DataFrame
    ) -> None:
        # val_iou: p=0.250 ≥ 0.05 → stamp colour must be muted grey
        import matplotlib.colors as mc
        # isolate only the IoU panel
        row_stats = {"val_iou": paired_results.loc["val_iou"]}
        fig = plot_paired_slopegraph(
            paired_runs, [("val_iou", "IoU")],
            arch_a="A", arch_b="B", seeds=_SEEDS, palette=_PALETTE,
            row_stats=row_stats, alpha=0.05,
        )
        stamp_colours = [
            mc.to_hex(t.get_color())
            for ax in fig.axes for t in ax.texts
            if "p = " in t.get_text()
        ]
        assert stamp_colours, "IoU stamp must exist"
        assert all(c.lower() == "#999999" for c in stamp_colours), (
            "Non-significant stamp must be grey (#999999)"
        )


# ---------------------------------------------------------------------------
# row_stats integration — forest p and d_z in marginal annotation
# ---------------------------------------------------------------------------

class TestForestRowStats:
    _SPECS = [
        ("Dice", "val_dice", False),
        ("IoU", "val_iou", False),
        ("Gen-gap", "gen_gap", True),
    ]

    def _make(self, paired_results: pd.DataFrame, **kw: Any) -> Figure:
        return plot_paired_forest(
            paired_results, self._SPECS,
            arch_a="A", arch_b="B", n=4, palette=_PALETTE,
            alpha=0.05, **kw,
        )

    def test_marginal_text_contains_p(self, paired_results: pd.DataFrame) -> None:
        # p appears in the italic stat line below the CI bracket
        fig = self._make(paired_results)
        stat_texts = [t.get_text() for t in fig.axes[0].texts
                      if "Wilcoxon p" in t.get_text()]
        assert len(stat_texts) >= len(self._SPECS), (
            "Each row must have an italic stat line containing 'Wilcoxon p'"
        )

    def test_marginal_text_contains_dz(self, paired_results: pd.DataFrame) -> None:
        # d_z appears in the italic stat line below the CI bracket
        fig = self._make(paired_results)
        stat_texts = [t.get_text() for t in fig.axes[0].texts
                      if "d_z = " in t.get_text()]
        assert len(stat_texts) >= len(self._SPECS), (
            "Each row must have an italic stat line containing 'd_z ='"
        )

    def test_missing_stat_cols_no_crash(self) -> None:
        # results without wilcoxon_p / cohen_dz — should still render without error
        df = pd.DataFrame({
            "delta_a_minus_b": [0.005, -0.002],
            "boot_lo": [0.001, -0.006],
            "boot_hi": [0.009, 0.002],
        }, index=pd.Index(["val_dice", "gen_gap"], name="metric"))
        fig = plot_paired_forest(
            df, [("Dice", "val_dice", False), ("Gen-gap", "gen_gap", True)],
            arch_a="A", arch_b="B", n=4, palette=_PALETTE, alpha=0.05,
        )
        assert isinstance(fig, Figure)

    def test_significant_row_uses_winner_colour(
        self, paired_results: pd.DataFrame
    ) -> None:
        # val_dice: p=0.031 < 0.05, delta > 0 → stat-line colour = palette["A"]
        import matplotlib.colors as mc
        fig = self._make(paired_results)
        dice_stamps = [t for t in fig.axes[0].texts
                       if "Wilcoxon p = 0.031" in t.get_text()]
        assert dice_stamps, "Dice row stat line must exist and contain 'Wilcoxon p = 0.031'"
        c = mc.to_hex(dice_stamps[0].get_color()).lower()
        assert c == _PALETTE["A"].lower(), (
            "Significant winner row must use palette[arch_a] colour"
        )

    def test_nonsignificant_row_uses_grey(
        self, paired_results: pd.DataFrame
    ) -> None:
        # val_iou: p=0.250 ≥ 0.05 → stat-line colour = #999999
        import matplotlib.colors as mc
        fig = self._make(paired_results)
        iou_stamps = [t for t in fig.axes[0].texts
                      if "Wilcoxon p = 0.250" in t.get_text()]
        assert iou_stamps, "IoU row stat line must exist and contain 'Wilcoxon p = 0.250'"
        c = mc.to_hex(iou_stamps[0].get_color()).lower()
        assert c == "#999999", "Non-significant row must use grey (#999999)"
