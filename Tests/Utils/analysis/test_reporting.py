"""Unit tests for SkiNet.Utils.analysis.reporting."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from SkiNet.Utils.analysis.reporting import show_run_table, show_gain_table
from SkiNet.Utils.analysis.schema import ARCH, SEED
from SkiNet.Utils.analysis.reporting import show_comparison_table, show_family_verdicts

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_E2 = np.array([0.0081, 0.0009, 0.0010, 0.0078, 0.0009, 0.0042, 0.0008, -0.0007, 0.0049, -0.0026])


def _results() -> pd.DataFrame:
    """Minimal results DataFrame matching build_comparison_table output."""
    mean_d = float(_E2.mean())
    sd = float(_E2.std(ddof=1))
    rows = []
    for i, m in enumerate(["val_dice_max", "val_dice_tail_mean", "val_iou_max",
                           "generalization_gap_final", "samples_per_sec"]):
        sign = 1 if i < 3 else -1
        rows.append({
            "metric": m,
            "a_mean": 0.80 + mean_d / 2,
            "b_mean": 0.80 - mean_d / 2,
            "delta_a_minus_b": mean_d * sign,
            "boot_lo": mean_d * sign - 0.002,
            "boot_hi": mean_d * sign + 0.003,
            "wilcoxon_p": 0.037 if m == "val_dice_tail_mean" else (0.002 if m == "samples_per_sec" else 0.50),
            "bonferroni_sig": m in {"val_dice_tail_mean", "samples_per_sec"},
            "cohen_dz": mean_d / sd,
            "dz_lo": mean_d / sd - 0.3,
            "dz_hi": mean_d / sd + 0.3,
        })
    return pd.DataFrame(rows).set_index("metric")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _runs(n_seeds: int = 3) -> pd.DataFrame:
    """Minimal long-format runs table."""
    archs = ["classical", "attention_gate"]
    records = []
    rng = np.random.default_rng(0)
    for arch in archs:
        for seed in range(n_seeds):
            records.append({
                ARCH: arch,
                SEED: seed,
                "val_dice_max": float(rng.uniform(0.80, 0.90)),
                "val_dice_tail_mean": float(rng.uniform(0.78, 0.88)),
                "val_dice_tail_std": float(rng.uniform(0.001, 0.01)),
                "val_iou_max": float(rng.uniform(0.70, 0.82)),
                "generalization_gap_final": float(rng.uniform(-0.02, 0.02)),
                "samples_per_sec": float(rng.uniform(40, 60)),
                "duration_min": float(rng.uniform(30, 90)),
            })
    return pd.DataFrame(records)


def _per_seed(n: int = 5) -> pd.DataFrame:
    """Minimal per-seed threshold-sweep result table."""
    rng = np.random.default_rng(42)
    base = rng.uniform(0.80, 0.85, size=n)
    return pd.DataFrame({
        "seed": np.arange(n),
        "val_best_dice_at_threshold": base + rng.uniform(0.001, 0.01, size=n),
        "val_dice": base,
    })


def _gain(per_seed: pd.DataFrame, wilcoxon_p: float = 0.02) -> dict:
    """Minimal gain dict matching paired_gain_stats output."""
    deltas = (per_seed["val_best_dice_at_threshold"] - per_seed["val_dice"]).to_numpy()
    n = len(deltas)
    return {
        "mean": float(deltas.mean()),
        "ci_lo": float(deltas.mean() - 0.003),
        "ci_hi": float(deltas.mean() + 0.004),
        "wilcoxon_p": wilcoxon_p,
        "cohen_dz": float(deltas.mean() / deltas.std(ddof=1)),
        "n_positive": int((deltas > 0).sum()),
        "n": n,
        "per_seed_val_dice_gain": deltas.tolist(),
    }


# ---------------------------------------------------------------------------
# show_run_table
# ---------------------------------------------------------------------------

class TestShowRunTable:

    def test_calls_display_once(self) -> None:
        with patch("SkiNet.Utils.analysis.reporting.display") as mock_display:
            show_run_table(_runs())
            mock_display.assert_called_once()

    def test_default_columns_present(self) -> None:
        captured: dict = {}

        def _cap(df: pd.DataFrame) -> None:
            captured["df"] = df

        with patch("SkiNet.Utils.analysis.reporting.display", side_effect=_cap):
            show_run_table(_runs())

        df = captured["df"]
        # ARCH and SEED must appear; at least one metric column too
        assert ARCH in df.columns
        assert SEED in df.columns
        assert "val_dice_max" in df.columns

    def test_custom_columns_respected(self) -> None:
        captured: dict = {}

        def _cap(df: pd.DataFrame) -> None:
            captured["df"] = df

        cols = [ARCH, SEED, "val_dice_max"]
        with patch("SkiNet.Utils.analysis.reporting.display", side_effect=_cap):
            show_run_table(_runs(), columns=cols)

        assert list(captured["df"].columns) == cols

    def test_default_sort_by_arch_then_seed(self) -> None:
        captured: dict = {}

        def _cap(df: pd.DataFrame) -> None:
            captured["df"] = df

        with patch("SkiNet.Utils.analysis.reporting.display", side_effect=_cap):
            show_run_table(_runs())

        df = captured["df"]
        archs = df[ARCH].tolist()
        # All rows for the first arch label should form a contiguous block
        first_switch = next(
            (i for i in range(1, len(archs)) if archs[i] != archs[i - 1]), None
        )
        assert first_switch is not None, "Expected two arch groups"
        assert archs[:first_switch] == [archs[0]] * first_switch

    def test_custom_sort_by(self) -> None:
        captured: dict = {}

        def _cap(df: pd.DataFrame) -> None:
            captured["df"] = df

        with patch("SkiNet.Utils.analysis.reporting.display", side_effect=_cap):
            show_run_table(_runs(), sort_by=(SEED, ARCH))

        seeds = captured["df"][SEED].tolist()
        assert seeds == sorted(seeds), "Rows should be sorted by seed first"

    def test_index_reset(self) -> None:
        """Displayed DataFrame index must start at 0 (reset_index applied)."""
        captured: dict = {}

        def _cap(df: pd.DataFrame) -> None:
            captured["df"] = df

        with patch("SkiNet.Utils.analysis.reporting.display", side_effect=_cap):
            show_run_table(_runs())

        assert list(captured["df"].index) == list(range(len(captured["df"])))

    def test_extra_columns_in_default_list_but_absent_from_frame_are_dropped(self) -> None:
        """Columns listed in _DEFAULT_RUN_COLS but absent from the frame are silently dropped."""
        runs = _runs().drop(columns=["duration_min", "samples_per_sec"])
        with patch("SkiNet.Utils.analysis.reporting.display") as mock_display:
            show_run_table(runs)  # must not raise
            mock_display.assert_called_once()

    def test_single_row_frame(self) -> None:
        with patch("SkiNet.Utils.analysis.reporting.display") as mock_display:
            show_run_table(_runs().iloc[:1])
            mock_display.assert_called_once()

    def test_empty_frame(self) -> None:
        empty = _runs().iloc[:0]
        captured: dict = {}

        def _cap(df: pd.DataFrame) -> None:
            captured["df"] = df

        with patch("SkiNet.Utils.analysis.reporting.display", side_effect=_cap):
            show_run_table(empty)

        assert len(captured["df"]) == 0


# ---------------------------------------------------------------------------
# show_gain_table
# ---------------------------------------------------------------------------

class TestShowGainTable:

    def test_calls_display_once(self) -> None:
        ps = _per_seed()
        with patch("SkiNet.Utils.analysis.reporting.display") as mock_display:
            show_gain_table(ps, _gain(ps))
            mock_display.assert_called_once()

    def test_displayed_dataframe_has_wins_column(self) -> None:
        captured: dict = {}

        def _cap(df: pd.DataFrame) -> None:
            captured["df"] = df

        ps = _per_seed()
        with patch("SkiNet.Utils.analysis.reporting.display", side_effect=_cap):
            show_gain_table(ps, _gain(ps))

        assert "wins" in captured["df"].columns

    def test_wins_formatted_as_fraction(self) -> None:
        captured: dict = {}

        def _cap(df: pd.DataFrame) -> None:
            captured["df"] = df

        ps = _per_seed()
        g = _gain(ps)
        with patch("SkiNet.Utils.analysis.reporting.display", side_effect=_cap):
            show_gain_table(ps, g)

        wins_val = captured["df"]["wins"].iloc[0]
        assert "/" in wins_val, f"Expected 'n/N' format, got {wins_val!r}"
        left, right = wins_val.split("/")
        assert left.isdigit() and right.isdigit()

    def test_index_label_is_in_sample_dice_gain(self) -> None:
        captured: dict = {}

        def _cap(df: pd.DataFrame) -> None:
            captured["df"] = df

        ps = _per_seed()
        with patch("SkiNet.Utils.analysis.reporting.display", side_effect=_cap):
            show_gain_table(ps, _gain(ps))

        assert "in_sample_dice_gain" in captured["df"].index

    def test_reject_h0_printed_when_significant(self, capsys: pytest.CaptureFixture) -> None:
        ps = _per_seed()
        g = _gain(ps, wilcoxon_p=0.01)
        g["ci_lo"] = 0.001   # CI does not straddle zero
        with patch("SkiNet.Utils.analysis.reporting.display"):
            show_gain_table(ps, g, alpha=0.05)
        out = capsys.readouterr().out
        assert "REJECT H0" in out

    def test_fail_to_reject_printed_when_p_not_significant(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        ps = _per_seed()
        g = _gain(ps, wilcoxon_p=0.80)
        with patch("SkiNet.Utils.analysis.reporting.display"):
            show_gain_table(ps, g, alpha=0.05)
        out = capsys.readouterr().out
        assert "fail to reject H0" in out

    def test_fail_to_reject_when_ci_straddles_zero(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """p < alpha but CI spans zero → fail to reject (both conditions must hold)."""
        ps = _per_seed()
        g = _gain(ps, wilcoxon_p=0.01)
        g["ci_lo"] = -0.005   # straddles zero
        g["ci_hi"] = +0.005
        with patch("SkiNet.Utils.analysis.reporting.display"):
            show_gain_table(ps, g, alpha=0.05)
        out = capsys.readouterr().out
        assert "fail to reject H0" in out

    def test_alpha_boundary_p_equals_alpha_is_not_rejected(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """p == alpha is not strictly less-than → H0 must be retained."""
        ps = _per_seed()
        g = _gain(ps, wilcoxon_p=0.05)
        g["ci_lo"] = 0.001
        with patch("SkiNet.Utils.analysis.reporting.display"):
            show_gain_table(ps, g, alpha=0.05)
        out = capsys.readouterr().out
        assert "fail to reject H0" in out

    def test_per_seed_deltas_printed(self, capsys: pytest.CaptureFixture) -> None:
        ps = _per_seed()
        g = _gain(ps)
        with patch("SkiNet.Utils.analysis.reporting.display"):
            show_gain_table(ps, g)
        out = capsys.readouterr().out
        assert "per-seed" in out.lower()

    def test_sig_checkmark_when_significant(self) -> None:
        captured: dict = {}

        def _cap(df: pd.DataFrame) -> None:
            captured["df"] = df

        ps = _per_seed()
        g = _gain(ps, wilcoxon_p=0.01)
        with patch("SkiNet.Utils.analysis.reporting.display", side_effect=_cap):
            show_gain_table(ps, g, alpha=0.05)

        assert captured["df"]["sig"].iloc[0] == "✓"

    def test_sig_empty_when_not_significant(self) -> None:
        captured: dict = {}

        def _cap(df: pd.DataFrame) -> None:
            captured["df"] = df

        ps = _per_seed()
        g = _gain(ps, wilcoxon_p=0.80)
        with patch("SkiNet.Utils.analysis.reporting.display", side_effect=_cap):
            show_gain_table(ps, g, alpha=0.05)

        assert captured["df"]["sig"].iloc[0] == ""

    def test_displayed_dataframe_has_eight_columns(self) -> None:
        """7 base columns + wins extra_col = 8 total."""
        captured: dict = {}

        def _cap(df: pd.DataFrame) -> None:
            captured["df"] = df

        ps = _per_seed()
        with patch("SkiNet.Utils.analysis.reporting.display", side_effect=_cap):
            show_gain_table(ps, _gain(ps))

        assert len(captured["df"].columns) == 8

    def test_negative_delta_formatted_with_minus_sign(self) -> None:
        """Sign formatting holds when swept threshold performs worse than fixed τ=0.5."""
        captured: dict = {}

        def _cap(df: pd.DataFrame) -> None:
            captured["df"] = df

        ps = _per_seed()
        g = _gain(ps)
        g["mean"] = -0.0031
        g["ci_lo"] = -0.006
        g["ci_hi"] = -0.001
        with patch("SkiNet.Utils.analysis.reporting.display", side_effect=_cap):
            show_gain_table(ps, g)

        delta_val = captured["df"]["Δ (swept−0.5)"].iloc[0]
        assert delta_val.startswith("-"), f"Expected leading '-', got {delta_val!r}"

    def test_custom_alpha_changes_verdict(self, capsys: pytest.CaptureFixture) -> None:
        ps = _per_seed()
        g = _gain(ps, wilcoxon_p=0.04)
        g["ci_lo"] = 0.001  # CI above zero

        with patch("SkiNet.Utils.analysis.reporting.display"):
            show_gain_table(ps, g, alpha=0.05)   # 0.04 < 0.05 → reject
        out_reject = capsys.readouterr().out

        with patch("SkiNet.Utils.analysis.reporting.display"):
            show_gain_table(ps, g, alpha=0.01)   # 0.04 > 0.01 → retain
        out_retain = capsys.readouterr().out

        assert "REJECT H0" in out_reject
        assert "fail to reject H0" in out_retain


# ---------------------------------------------------------------------------
# show_comparison_table
# ---------------------------------------------------------------------------

class TestShowComparisonTable:
    def test_calls_display(self) -> None:
        with patch("SkiNet.Utils.analysis.reporting.display") as mock_display:
            show_comparison_table(_results())
            mock_display.assert_called_once()

    def test_displayed_dataframe_has_seven_columns(self) -> None:
        captured = {}

        def _capture(df: pd.DataFrame) -> None:
            captured['df'] = df
        with patch("SkiNet.Utils.analysis.reporting.display", side_effect=_capture):
            show_comparison_table(_results())
        assert len(captured['df'].columns) == 7

    def test_delta_column_formatted_with_sign(self) -> None:
        captured = {}

        def _capture(df: pd.DataFrame) -> None:
            captured['df'] = df
        with patch("SkiNet.Utils.analysis.reporting.display", side_effect=_capture):
            show_comparison_table(_results())
        delta_vals = captured['df']["Δ (AG−HE2)"].tolist()
        assert all(v.startswith(("+", "-")) for v in delta_vals)

    def test_ci_column_formatted_as_bracket(self) -> None:
        captured = {}

        def _capture(df: pd.DataFrame) -> None:
            captured['df'] = df
        with patch("SkiNet.Utils.analysis.reporting.display", side_effect=_capture):
            show_comparison_table(_results())
        ci_vals = captured['df']["95% BCa CI"].tolist()
        assert all(v.startswith("[") and v.endswith("]") for v in ci_vals)

    def test_sig_checkmark_for_significant_rows(self) -> None:
        captured = {}

        def _capture(df: pd.DataFrame) -> None:
            captured['df'] = df
        with patch("SkiNet.Utils.analysis.reporting.display", side_effect=_capture):
            show_comparison_table(_results())
        df = captured['df']
        assert df.loc["val_dice_tail_mean", "sig"] == "✓"
        assert df.loc["val_dice_max", "sig"] == ""

    def test_index_preserved(self) -> None:
        captured = {}

        def _capture(df: pd.DataFrame) -> None:
            captured['df'] = df
        with patch("SkiNet.Utils.analysis.reporting.display", side_effect=_capture):
            show_comparison_table(_results())
        assert "val_dice_tail_mean" in captured['df'].index


# ---------------------------------------------------------------------------
# show_family_verdicts
# ---------------------------------------------------------------------------

PRIMARY = "val_dice_tail_mean"
SECONDARY = ["val_dice_max", "val_iou_max", "generalization_gap_final"]


class TestShowFamilyVerdicts:
    def test_prints_primary_verdict(self, capsys: pytest.CaptureFixture) -> None:
        show_family_verdicts(_results(), PRIMARY, SECONDARY)
        out = capsys.readouterr().out
        assert "Primary" in out
        assert PRIMARY in out

    def test_primary_reject_when_p_below_alpha(self, capsys: pytest.CaptureFixture) -> None:
        show_family_verdicts(_results(), PRIMARY, SECONDARY, alpha=0.05)
        out = capsys.readouterr().out
        assert "REJECT H0" in out

    def test_primary_retain_when_p_above_alpha(self, capsys: pytest.CaptureFixture) -> None:
        show_family_verdicts(_results(), PRIMARY, SECONDARY, alpha=0.001)
        out = capsys.readouterr().out
        assert "retain H0" in out

    def test_prints_holm_section(self, capsys: pytest.CaptureFixture) -> None:
        show_family_verdicts(_results(), PRIMARY, SECONDARY)
        out = capsys.readouterr().out
        assert "Holm" in out
        assert "secondary family" in out

    def test_prints_throughput_section(self, capsys: pytest.CaptureFixture) -> None:
        show_family_verdicts(_results(), PRIMARY, SECONDARY)
        out = capsys.readouterr().out
        assert "samples_per_sec" in out

    def test_custom_throughput_metric(self, capsys: pytest.CaptureFixture) -> None:
        show_family_verdicts(_results(), PRIMARY, SECONDARY,
                             throughput_metric="val_iou_max")
        out = capsys.readouterr().out
        assert "val_iou_max" in out
