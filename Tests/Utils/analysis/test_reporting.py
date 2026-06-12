"""Unit tests for SkiNet.Utils.analysis.reporting."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

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
