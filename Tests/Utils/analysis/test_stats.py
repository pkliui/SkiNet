"""Unit tests for SkiNet.Utils.analysis.stats."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from SkiNet.Utils.analysis.stats import (
    _cohen_dz_scalar,
    bootstrap_paired_ci,
    build_comparison_table,
    holm_step_down,
    paired_metric_stats,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Actual E2 plateau-Dice paired differences (AG - HE2, seeds 100-109).
E2_PLATEAU = np.array(
    [0.0081, 0.0009, 0.0010, 0.0078, 0.0009, 0.0042, 0.0008, -0.0007, 0.0049, -0.0026]
)

# 10 symmetric differences around zero — true mean = 0, CI must straddle 0.
SYMMETRIC_ZERO = np.array([-0.010, -0.007, -0.004, -0.001, 0.000, 0.001, 0.004, 0.007, 0.009, 0.010])

# 10 large positive differences — CI should clearly exclude 0 from below.
ALL_POSITIVE = np.array([0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.085, 0.090, 0.095])

_RNG = 42
_N = 10_000


# ---------------------------------------------------------------------------
# _cohen_dz_scalar
# ---------------------------------------------------------------------------

class TestCohenDzScalar:
    def test_known_value(self) -> None:
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = 3.0 / np.std(d, ddof=1)
        assert math.isclose(_cohen_dz_scalar(d), expected, rel_tol=1e-9)

    def test_zero_mean(self) -> None:
        d = np.array([-1.0, 0.0, 1.0])
        assert _cohen_dz_scalar(d) == pytest.approx(0.0, abs=1e-12)

    def test_degenerate_sd_zero_returns_nan(self) -> None:
        d = np.array([0.5, 0.5, 0.5])
        assert math.isnan(_cohen_dz_scalar(d))

    def test_negative_mean(self) -> None:
        d = np.array([-3.0, -2.0, -1.0])
        assert _cohen_dz_scalar(d) < 0

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_single_element_sd_zero(self) -> None:
        # std(ddof=1) of a length-1 array is NaN in numpy → guard triggers → NaN.
        # numpy warns about the ddof<=0 divide; that NaN is exactly what we assert on.
        d = np.array([1.0])
        assert math.isnan(_cohen_dz_scalar(d))


# ---------------------------------------------------------------------------
# bootstrap_paired_ci
# ---------------------------------------------------------------------------

class TestBootstrapPairedCi:
    def test_returns_tuple_of_two_floats(self) -> None:
        lo, hi = bootstrap_paired_ci(E2_PLATEAU, n_resamples=_N, random_state=_RNG)
        assert isinstance(lo, float) and isinstance(hi, float)

    def test_lo_less_than_hi(self) -> None:
        lo, hi = bootstrap_paired_ci(E2_PLATEAU, n_resamples=_N, random_state=_RNG)
        assert lo < hi

    def test_e2_plateau_ci_excludes_zero(self) -> None:
        lo, hi = bootstrap_paired_ci(E2_PLATEAU, n_resamples=_N, random_state=_RNG)
        assert lo > 0.0

    def test_symmetric_zero_ci_straddles_zero(self) -> None:
        lo, hi = bootstrap_paired_ci(SYMMETRIC_ZERO, n_resamples=_N, random_state=_RNG)
        assert lo < 0.0 < hi

    def test_all_positive_ci_excludes_zero(self) -> None:
        lo, hi = bootstrap_paired_ci(ALL_POSITIVE, n_resamples=_N, random_state=_RNG)
        assert lo > 0.0

    def test_ci_contains_sample_mean(self) -> None:
        lo, hi = bootstrap_paired_ci(E2_PLATEAU, n_resamples=_N, random_state=_RNG)
        assert lo <= E2_PLATEAU.mean() <= hi

    def test_reproducible_with_same_seed(self) -> None:
        ci1 = bootstrap_paired_ci(E2_PLATEAU, n_resamples=_N, random_state=0)
        ci2 = bootstrap_paired_ci(E2_PLATEAU, n_resamples=_N, random_state=0)
        assert ci1 == ci2

    def test_different_seeds_may_differ(self) -> None:
        ci1 = bootstrap_paired_ci(E2_PLATEAU, n_resamples=_N, random_state=0)
        ci2 = bootstrap_paired_ci(E2_PLATEAU, n_resamples=_N, random_state=999)
        assert ci1 != ci2

    def test_accepts_generator(self) -> None:
        rng = np.random.default_rng(7)
        lo, hi = bootstrap_paired_ci(E2_PLATEAU, n_resamples=_N, random_state=rng)
        assert lo < hi

    def test_wider_ci_at_99_than_90(self) -> None:
        lo90, hi90 = bootstrap_paired_ci(E2_PLATEAU, n_resamples=_N, confidence_level=0.90, random_state=_RNG)
        lo99, hi99 = bootstrap_paired_ci(E2_PLATEAU, n_resamples=_N, confidence_level=0.99, random_state=_RNG)
        assert (hi99 - lo99) > (hi90 - lo90)

    def test_no_nan_on_real_data(self) -> None:
        lo, hi = bootstrap_paired_ci(E2_PLATEAU, n_resamples=_N, random_state=_RNG)
        assert not math.isnan(lo) and not math.isnan(hi)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.filterwarnings("ignore::scipy.stats.DegenerateDataWarning")
    def test_propagates_value_error(self) -> None:
        # All identical → every resample has SD=0 → BCa undefined → ValueError.
        # scipy emits the degenerate-distribution warnings en route to the NaN CI
        # that _bca_ci converts into the ValueError we assert on here.
        d = np.ones(10)
        with pytest.raises(ValueError, match="BCa CI is NaN"):
            bootstrap_paired_ci(d, n_resamples=500, random_state=_RNG)


# ---------------------------------------------------------------------------
# Shared helpers for paired_metric_stats / holm_step_down
# ---------------------------------------------------------------------------

def _long_df(d: np.ndarray = E2_PLATEAU, metric: str = "val_dice") -> pd.DataFrame:
    """Build a long-format runs table from a 1-D array of paired differences."""
    rows = []
    for i, diff in enumerate(d):
        rows.append({"seed": i, "arch": "A", metric: 0.80 + diff / 2})
        rows.append({"seed": i, "arch": "B", metric: 0.80 - diff / 2})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# paired_metric_stats
# ---------------------------------------------------------------------------

class TestPairedMetricStats:
    def _stat(self, d: np.ndarray = E2_PLATEAU, **kwargs: Any) -> dict:
        return paired_metric_stats(
            _long_df(d), "val_dice", arch_a="A", arch_b="B",
            n_resamples=500, random_state=0, **kwargs,
        )

    def test_returns_dict(self) -> None:
        assert isinstance(self._stat(), dict)

    def test_expected_keys_present(self) -> None:
        keys = self._stat().keys()
        for k in ("metric", "a_mean", "b_mean", "delta_a_minus_b",
                  "boot_lo", "boot_hi", "wilcoxon_p", "bonferroni_sig",
                  "cohen_dz", "sign_n_a", "sign_p_2tail"):
            assert k in keys

    def test_means_correct(self) -> None:
        stat = self._stat()
        assert stat["a_mean"] == pytest.approx(0.80 + E2_PLATEAU.mean() / 2, rel=1e-6)
        assert stat["b_mean"] == pytest.approx(0.80 - E2_PLATEAU.mean() / 2, rel=1e-6)

    def test_delta_equals_mean_of_differences(self) -> None:
        stat = self._stat()
        assert stat["delta_a_minus_b"] == pytest.approx(E2_PLATEAU.mean(), rel=1e-6)

    def test_wilcoxon_p_in_unit_interval(self) -> None:
        stat = self._stat()
        assert 0.0 <= stat["wilcoxon_p"] <= 1.0

    def test_bonferroni_sig_true_when_p_below_threshold(self) -> None:
        # Use n_corrections=1 so threshold=0.05; E2 plateau p=0.037 is significant.
        stat = self._stat(n_corrections=1)
        assert stat["bonferroni_sig"] is True

    def test_bonferroni_sig_false_when_strict_family(self) -> None:
        # n_corrections=100 → threshold=0.0005; 0.037 should not pass.
        stat = self._stat(n_corrections=100)
        assert stat["bonferroni_sig"] is False

    def test_sign_n_a_counts_positive_diffs(self) -> None:
        stat = self._stat(higher_is_better=True)
        expected = int((E2_PLATEAU > 0).sum())
        assert stat["sign_n_a"] == expected

    def test_sign_n_a_counts_negative_diffs_when_lower_is_better(self) -> None:
        stat = self._stat(higher_is_better=False)
        expected = int((E2_PLATEAU < 0).sum())
        assert stat["sign_n_a"] == expected

    def test_cohen_dz_correct_sign(self) -> None:
        assert self._stat()["cohen_dz"] > 0  # E2 plateau: positive mean

    def test_seeds_parameter_filters_rows(self) -> None:
        stat_full = self._stat()
        stat_half = paired_metric_stats(
            _long_df(), "val_dice", arch_a="A", arch_b="B",
            seeds=list(range(8)), n_resamples=500, random_state=0,
        )
        # Fewer seeds → different (typically wider) CI
        assert stat_full["a_mean"] != stat_half["a_mean"]

    def test_metric_name_preserved(self) -> None:
        assert self._stat()["metric"] == "val_dice"


# ---------------------------------------------------------------------------
# holm_step_down
# ---------------------------------------------------------------------------

class TestHolmStepDown:
    def test_returns_dataframe(self) -> None:
        result = holm_step_down({"a": 0.01, "b": 0.04, "c": 0.20})
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self) -> None:
        result = holm_step_down({"a": 0.01})
        assert {"p", "threshold", "reject"}.issubset(result.columns)

    def test_sorted_ascending_by_p(self) -> None:
        result = holm_step_down({"a": 0.20, "b": 0.01, "c": 0.05})
        assert result["p"].tolist() == sorted(result["p"].tolist())

    def test_all_significant(self) -> None:
        result = holm_step_down({"a": 0.001, "b": 0.002, "c": 0.003}, alpha=0.05)
        assert result["reject"].all()

    def test_none_significant(self) -> None:
        result = holm_step_down({"a": 0.50, "b": 0.60, "c": 0.70}, alpha=0.05)
        assert not result["reject"].any()

    def test_step_down_stops_at_first_failure(self) -> None:
        # k=3: thresholds are 0.0167, 0.025, 0.05
        # p-values: 0.01 (pass), 0.03 (fail at 0.025), 0.04 (should also be False)
        result = holm_step_down({"a": 0.01, "b": 0.03, "c": 0.04}, alpha=0.05)
        assert bool(result.loc["a", "reject"]) is True
        assert bool(result.loc["b", "reject"]) is False
        assert bool(result.loc["c", "reject"]) is False

    def test_single_hypothesis_threshold_equals_alpha(self) -> None:
        result = holm_step_down({"only": 0.03}, alpha=0.05)
        assert result.loc["only", "threshold"] == pytest.approx(0.05)

    def test_thresholds_increase_with_rank(self) -> None:
        # Smallest p is held to the strictest (smallest) threshold, so the
        # per-rank threshold alpha / (k - i) grows as rank i increases.
        result = holm_step_down({"a": 0.01, "b": 0.02, "c": 0.03}, alpha=0.06)
        thresholds = result["threshold"].tolist()
        # threshold at rank 0: 0.06/3=0.02, rank 1: 0.06/2=0.03, rank 2: 0.06/1=0.06
        assert thresholds[0] < thresholds[1] < thresholds[2]


# ---------------------------------------------------------------------------
# build_comparison_table
# ---------------------------------------------------------------------------

class TestBuildComparisonTable:
    SPEC = [
        ("val_dice", True, 2),
        ("val_iou", True, 2),
    ]

    def _df(self) -> pd.DataFrame:
        d1 = E2_PLATEAU
        d2 = E2_PLATEAU * 0.5
        rows = []
        for i, (v1, v2) in enumerate(zip(d1, d2)):
            rows += [
                {"seed": i, "arch": "A", "val_dice": 0.80 + v1 / 2, "val_iou": 0.70 + v2 / 2},
                {"seed": i, "arch": "B", "val_dice": 0.80 - v1 / 2, "val_iou": 0.70 - v2 / 2},
            ]
        return pd.DataFrame(rows)

    def test_returns_dataframe(self) -> None:
        result = build_comparison_table(self._df(), self.SPEC, arch_a="A", arch_b="B",
                                        n_resamples=200, random_state=0)
        assert isinstance(result, pd.DataFrame)

    def test_index_is_metric_names(self) -> None:
        result = build_comparison_table(self._df(), self.SPEC, arch_a="A", arch_b="B",
                                        n_resamples=200, random_state=0)
        assert list(result.index) == ["val_dice", "val_iou"]

    def test_one_row_per_metric(self) -> None:
        result = build_comparison_table(self._df(), self.SPEC, arch_a="A", arch_b="B",
                                        n_resamples=200, random_state=0)
        assert len(result) == 2

    def test_generic_column_names_present(self) -> None:
        result = build_comparison_table(self._df(), self.SPEC, arch_a="A", arch_b="B",
                                        n_resamples=200, random_state=0)
        for col in ("a_mean", "b_mean", "delta_a_minus_b", "wilcoxon_p", "cohen_dz"):
            assert col in result.columns

    def test_n_corrections_respected(self) -> None:
        # k=1 → Bonferroni threshold = 0.05; k=100 → threshold = 0.0005
        spec_k1 = [("val_dice", True, 1)]
        spec_k100 = [("val_dice", True, 100)]
        r1 = build_comparison_table(self._df(), spec_k1, arch_a="A", arch_b="B",
                                    n_resamples=200, random_state=0)
        r100 = build_comparison_table(self._df(), spec_k100, arch_a="A", arch_b="B",
                                      n_resamples=200, random_state=0)
        # With k=1 the E2 plateau (p≈0.037) should be significant; k=100 should not.
        assert bool(r1.loc["val_dice", "bonferroni_sig"]) is True
        assert bool(r100.loc["val_dice", "bonferroni_sig"]) is False
