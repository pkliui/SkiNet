"""Tests for SkiNet.Utils.analysis.batch_sweep."""

from __future__ import annotations
import sqlite3
from pathlib import Path
from collections.abc import Sequence
import numpy as np
import pandas as pd
import pytest

from SkiNet.Utils.analysis.batch_sweep import (
    add_max_efficiency,
    add_scaling_metrics,
    build_run_frame,
    get_rule_outlier_steps,
    gpu_summary,
    load_active_runs,
    load_experiment,
    load_metric_series,
    make_placeholder,
    mark_outliers,
    plateau_batch_sizes,
    recommendation_facts,
    throughput_summary,
)
from SkiNet.Utils.analysis.schema import TIDY_COLS


# ---------------------------------------------------------------------------
# In-memory SQLite helpers
# ---------------------------------------------------------------------------

def _make_db(batch_sizes: Sequence[int] = (8, 16),
             steps_per_bs: int = 50,
             run_prefix: str = "run") -> sqlite3.Connection:
    """Create a minimal MLflow-shaped in-memory SQLite DB."""
    con = sqlite3.connect(":memory:")
    con.executescript("""
        CREATE TABLE runs (
            run_uuid TEXT PRIMARY KEY,
            lifecycle_stage TEXT
        );
        CREATE TABLE params (
            run_uuid TEXT,
            key TEXT,
            value TEXT
        );
        CREATE TABLE metrics (
            run_uuid TEXT,
            key TEXT,
            value REAL,
            step INTEGER,
            timestamp INTEGER
        );
    """)
    rng = np.random.default_rng(0)
    for bs in batch_sizes:
        uuid = f"{run_prefix}-bs{bs}"
        con.execute("INSERT INTO runs VALUES (?, 'active')", (uuid,))
        con.execute("INSERT INTO params VALUES (?, 'batch_size', ?)", (uuid, str(bs)))
        base_sps = 120 + 5 * bs
        for step in range(steps_per_bs):
            for key, val in [
                ("perf/samples_per_sec", float(rng.normal(base_sps, 3))),
                ("perf/time_per_step_ms", float(bs / base_sps * 1000)),
                ("system/gpu_mem_allocated_gb", float(0.5 + bs * 0.05)),
                ("system/gpu_util_percent", float(min(99, 40 + bs * 0.5))),
                ("train_loss_step", float(0.5 * np.exp(-step / 200))),
                ("epoch", float(step // max(1, steps_per_bs // 10))),
            ]:
                con.execute(
                    "INSERT INTO metrics VALUES (?, ?, ?, ?, ?)",
                    (uuid, key, val, step, step),
                )
    con.commit()
    return con


def _minimal_tidy(batch_sizes: Sequence[int] = (4, 8, 16),
                  n_steps: int = 100,
                  experiment: str = "no_aug") -> pd.DataFrame:
    """Build a minimal tidy DataFrame for unit-testing summary functions."""
    rng = np.random.default_rng(1)
    rows = []
    for bs in batch_sizes:
        base = 100 + 5 * bs
        for step in range(n_steps):
            rows.append({
                "experiment": experiment,
                "batch_size": bs,
                "run_uuid": f"r-{bs}",
                "step": step,
                "samples_per_sec": float(rng.normal(base, 4)),
                "time_per_step_ms": float(bs / base * 1000),
                "epoch_idx": float(step // 10),
                "gpu_mem_gb": float(0.5 + bs * 0.05),
                "gpu_util_pct": float(40 + bs * 0.5),
                "train_loss": float(0.5 * np.exp(-step / 200)),
                "is_outlier": False,
                "outlier_reason": "",
            })
    return pd.DataFrame(rows, columns=TIDY_COLS)


# ---------------------------------------------------------------------------
# load_active_runs
# ---------------------------------------------------------------------------

class TestLoadActiveRuns:
    def test_returns_expected_columns(self) -> None:
        con = _make_db()
        result = load_active_runs(con)
        assert set(result.columns) == {"run_uuid", "batch_size"}
        con.close()

    def test_batch_size_is_int(self) -> None:
        con = _make_db()
        result = load_active_runs(con)
        assert result["batch_size"].dtype == int
        con.close()

    def test_filters_to_allowed_batch_sizes(self) -> None:
        con = _make_db(batch_sizes=[8, 16, 32])
        result = load_active_runs(con, batch_sizes=[8, 16])
        assert set(result["batch_size"]) == {8, 16}
        con.close()

    def test_excludes_deleted_runs(self) -> None:
        con = _make_db(batch_sizes=[8])
        con.execute("UPDATE runs SET lifecycle_stage = 'deleted' WHERE run_uuid = 'run-bs8'")
        result = load_active_runs(con, batch_sizes=[8])
        assert len(result) == 0
        con.close()


# ---------------------------------------------------------------------------
# load_metric_series
# ---------------------------------------------------------------------------

class TestLoadMetricSeries:
    def test_returns_step_value_timestamp(self) -> None:
        con = _make_db(batch_sizes=[8], steps_per_bs=5)
        uuid = "run-bs8"
        result = load_metric_series(con, uuid, "perf/samples_per_sec")
        assert set(result.columns) >= {"step", "value", "timestamp"}
        con.close()

    def test_ordered_by_step(self) -> None:
        con = _make_db(batch_sizes=[8], steps_per_bs=20)
        result = load_metric_series(con, "run-bs8", "perf/samples_per_sec")
        assert list(result["step"]) == sorted(result["step"].tolist())
        con.close()

    def test_unknown_key_returns_empty(self) -> None:
        con = _make_db(batch_sizes=[8], steps_per_bs=5)
        result = load_metric_series(con, "run-bs8", "nonexistent/key")
        assert result.empty
        con.close()


# ---------------------------------------------------------------------------
# build_run_frame
# ---------------------------------------------------------------------------

class TestBuildRunFrame:
    def test_output_columns_match_tidy_cols(self) -> None:
        con = _make_db(batch_sizes=[8], steps_per_bs=20)
        result = build_run_frame(con, "run-bs8", 8, "no_aug")
        assert list(result.columns) == TIDY_COLS
        con.close()

    def test_experiment_and_batch_size_set(self) -> None:
        con = _make_db(batch_sizes=[16], steps_per_bs=10)
        result = build_run_frame(con, "run-bs16", 16, "with_aug")
        assert (result["experiment"] == "with_aug").all()
        assert (result["batch_size"] == 16).all()
        con.close()

    def test_is_outlier_false_by_default(self) -> None:
        con = _make_db(batch_sizes=[8], steps_per_bs=10)
        result = build_run_frame(con, "run-bs8", 8, "no_aug")
        assert not result["is_outlier"].any()
        con.close()

    def test_empty_on_missing_run_uuid(self) -> None:
        con = _make_db(batch_sizes=[8], steps_per_bs=5)
        result = build_run_frame(con, "nonexistent-uuid", 8, "no_aug")
        assert result.empty
        con.close()


# ---------------------------------------------------------------------------
# get_rule_outlier_steps
# ---------------------------------------------------------------------------

class TestGetRuleOutlierSteps:
    def test_step0_always_included(self) -> None:
        result = get_rule_outlier_steps(spe=50)
        assert 0 in result
        assert result[0] == "step0"

    def test_epoch_last_step_included(self) -> None:
        result = get_rule_outlier_steps(spe=50, max_epochs=10)
        assert 49 in result
        assert result[49] == "epoch_last_high"

    def test_epoch_first_step_included(self) -> None:
        result = get_rule_outlier_steps(spe=50, max_epochs=10)
        assert 50 in result
        assert result[50] == "epoch_first_low"

    def test_returns_dict(self) -> None:
        assert isinstance(get_rule_outlier_steps(spe=10), dict)


# ---------------------------------------------------------------------------
# mark_outliers
# ---------------------------------------------------------------------------

class TestMarkOutliers:
    def _frame(self, n: int = 120) -> pd.DataFrame:
        df = _minimal_tidy(batch_sizes=[8], n_steps=n).copy()
        df["is_outlier"] = False
        df["outlier_reason"] = ""
        return df

    def test_step0_flagged(self) -> None:
        df = mark_outliers(self._frame())
        assert df.loc[df["step"] == 0, "is_outlier"].all()

    def test_step0_reason_is_step0(self) -> None:
        df = mark_outliers(self._frame())
        assert (df.loc[df["step"] == 0, "outlier_reason"] == "step0").all()

    def test_clean_rows_not_flagged(self) -> None:
        df = mark_outliers(self._frame())
        n_clean = (~df["is_outlier"]).sum()
        assert n_clean > 0

    def test_empty_frame_returns_empty(self) -> None:
        empty = pd.DataFrame(columns=TIDY_COLS)
        result = mark_outliers(empty)
        assert result.empty

    def test_spe_stored_in_attrs(self) -> None:
        df = mark_outliers(self._frame(n=100))
        assert "spe" in df.attrs
        assert df.attrs["spe"] >= 1

    def test_iqr_outlier_flagged(self) -> None:
        df = _minimal_tidy(batch_sizes=[8], n_steps=200).copy()
        # inject a clear spike far outside IQR
        df.loc[50, "samples_per_sec"] = 1e6
        df = mark_outliers(df)
        assert df.loc[50, "is_outlier"]
        assert df.loc[50, "outlier_reason"] == "iqr"


# ---------------------------------------------------------------------------
# load_experiment
# ---------------------------------------------------------------------------

class TestLoadExperiment:
    def test_returns_empty_when_path_none(self) -> None:
        result = load_experiment("no_aug", None)
        assert result.empty

    def test_returns_empty_when_path_missing(self, tmp_path: Path) -> None:
        result = load_experiment("no_aug", tmp_path / "nonexistent.db")
        assert result.empty

    def test_loads_real_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        con = sqlite3.connect(str(db_path))
        # copy the in-memory DB structure to a file-backed DB
        _make_db(batch_sizes=[8, 16], steps_per_bs=30)
        # rebuild directly as a file DB
        con.executescript("""
            CREATE TABLE runs (run_uuid TEXT PRIMARY KEY, lifecycle_stage TEXT);
            CREATE TABLE params (run_uuid TEXT, key TEXT, value TEXT);
            CREATE TABLE metrics (run_uuid TEXT, key TEXT, value REAL,
                                  step INTEGER, timestamp INTEGER);
        """)
        rng = np.random.default_rng(0)
        for bs in [8, 16]:
            uuid = f"r-{bs}"
            con.execute("INSERT INTO runs VALUES (?, 'active')", (uuid,))
            con.execute("INSERT INTO params VALUES (?, 'batch_size', ?)", (uuid, str(bs)))
            for step in range(30):
                con.execute(
                    "INSERT INTO metrics VALUES (?, 'perf/samples_per_sec', ?, ?, ?)",
                    (uuid, float(rng.normal(120, 5)), step, step),
                )
        con.commit()
        con.close()
        result = load_experiment("no_aug", db_path, batch_sizes=[8, 16])
        assert not result.empty
        assert set(result["batch_size"].unique()) == {8, 16}
        assert list(result.columns) == TIDY_COLS


# ---------------------------------------------------------------------------
# make_placeholder
# ---------------------------------------------------------------------------

class TestMakePlaceholder:
    def test_returns_all_batch_sizes(self) -> None:
        df = make_placeholder("no_aug", batch_sizes=[4, 8])
        assert set(df["batch_size"].unique()) == {4, 8}

    def test_columns_match_tidy_cols(self) -> None:
        df = make_placeholder("no_aug", batch_sizes=[4])
        assert list(df.columns) == TIDY_COLS

    def test_outlier_marking_applied(self) -> None:
        df = make_placeholder("no_aug", batch_sizes=[8], max_epochs=3, n_train=80)
        assert df["is_outlier"].any(), "Placeholder must have at least step-0 outlier"

    def test_deterministic_output(self) -> None:
        df1 = make_placeholder("no_aug", batch_sizes=[8], max_epochs=2, n_train=80)
        df2 = make_placeholder("no_aug", batch_sizes=[8], max_epochs=2, n_train=80)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# throughput_summary
# ---------------------------------------------------------------------------

class TestThroughputSummary:
    def test_one_row_per_experiment_batch_size(self) -> None:
        df = _minimal_tidy(batch_sizes=[4, 8, 16])
        result = throughput_summary(df)
        assert len(result) == 3

    def test_excludes_outlier_rows(self) -> None:
        df = _minimal_tidy(batch_sizes=[8], n_steps=100)
        df.loc[0, "is_outlier"] = True
        df.loc[0, "samples_per_sec"] = 1e6  # extreme value
        summary = throughput_summary(df)
        assert summary.loc[0, "median"] < 1e5

    def test_n_clean_plus_n_outliers_equals_total(self) -> None:
        df = _minimal_tidy(batch_sizes=[8], n_steps=50)
        df.loc[:4, "is_outlier"] = True
        summary = throughput_summary(df)
        row = summary.iloc[0]
        assert row["n_clean"] + row["n_outliers"] == 50

    def test_p10_le_median_le_p90(self) -> None:
        df = _minimal_tidy(batch_sizes=[8])
        summary = throughput_summary(df)
        row = summary.iloc[0]
        assert row["p10"] <= row["median"] <= row["p90"]


# ---------------------------------------------------------------------------
# add_scaling_metrics
# ---------------------------------------------------------------------------

class TestAddScalingMetrics:
    def _summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "experiment": ["exp"] * 3,
            "batch_size": [4, 8, 16],
            "median": [100.0, 180.0, 200.0],
            "p10": [90.0, 170.0, 190.0],
            "p90": [110.0, 190.0, 210.0],
            "median_time_per_step_ms": [30.0, 40.0, 80.0],
        })

    def test_ref_bs_efficiency_is_100(self) -> None:
        result = add_scaling_metrics(self._summary(), ref_bs=4)
        row = result[result["batch_size"] == 4].iloc[0]
        assert row["efficiency_pct"] == pytest.approx(100.0)

    def test_perfect_scaling_at_bs8(self) -> None:
        result = add_scaling_metrics(self._summary(), ref_bs=4)
        row = result[result["batch_size"] == 8].iloc[0]
        assert row["perfect_scaling"] == pytest.approx(200.0)

    def test_missing_ref_bs_leaves_nan(self) -> None:
        s = self._summary()
        s = s[s["batch_size"] != 4].reset_index(drop=True)
        result = add_scaling_metrics(s, ref_bs=4)
        assert result["efficiency_pct"].isna().all()


# ---------------------------------------------------------------------------
# add_max_efficiency
# ---------------------------------------------------------------------------

class TestAddMaxEfficiency:
    def _summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "experiment": ["exp"] * 3,
            "batch_size": [4, 8, 16],
            "median": [100.0, 200.0, 160.0],
            "p10": [90.0] * 3,
            "p90": [110.0] * 3,
            "median_time_per_step_ms": [30.0] * 3,
        })

    def test_peak_batch_has_100_pct(self) -> None:
        result = add_max_efficiency(self._summary())
        peak_row = result.loc[result["median"].idxmax()]
        assert peak_row["eff_max_pct"] == pytest.approx(100.0)

    def test_all_values_between_0_and_100(self) -> None:
        result = add_max_efficiency(self._summary())
        assert (result["eff_max_pct"] >= 0).all()
        assert (result["eff_max_pct"] <= 100.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# plateau_batch_sizes
# ---------------------------------------------------------------------------

class TestPlateauBatchSizes:
    def _summary(self, effs: dict[int, float], exp: str = "exp") -> pd.DataFrame:
        rows = [{"experiment": exp, "batch_size": bs, "eff_max_pct": e}
                for bs, e in effs.items()]
        return pd.DataFrame(rows)

    def test_all_above_threshold(self) -> None:
        s = self._summary({4: 85.0, 8: 100.0, 16: 90.0})
        assert plateau_batch_sizes(s, threshold_pct=80.0) == [4, 8, 16]

    def test_some_below_threshold(self) -> None:
        s = self._summary({4: 60.0, 8: 100.0, 16: 90.0})
        assert plateau_batch_sizes(s, threshold_pct=80.0) == [8, 16]

    def test_must_pass_in_all_experiments(self) -> None:
        s = pd.concat([
            self._summary({4: 85.0, 8: 100.0}, exp="no_aug"),
            self._summary({4: 70.0, 8: 100.0}, exp="with_aug"),
        ], ignore_index=True)
        result = plateau_batch_sizes(s, threshold_pct=80.0)
        assert result == [8]

    def test_empty_when_none_qualify(self) -> None:
        s = self._summary({4: 50.0, 8: 60.0})
        assert plateau_batch_sizes(s, threshold_pct=80.0) == []


# ---------------------------------------------------------------------------
# gpu_summary
# ---------------------------------------------------------------------------

class TestGpuSummary:
    def test_one_row_per_experiment_batch_size(self) -> None:
        df = _minimal_tidy(batch_sizes=[4, 8])
        result = gpu_summary(df)
        assert len(result) == 2

    def test_excludes_outlier_rows(self) -> None:
        df = _minimal_tidy(batch_sizes=[8], n_steps=50)
        df.loc[0, "is_outlier"] = True
        df.loc[0, "gpu_util_pct"] = 999.0
        result = gpu_summary(df)
        assert result.iloc[0]["median_gpu_util_pct"] < 999.0

    def test_peak_mem_ge_median_mem(self) -> None:
        df = _minimal_tidy(batch_sizes=[8])
        result = gpu_summary(df)
        row = result.iloc[0]
        assert row["peak_gpu_mem_gb"] >= row["median_gpu_mem_gb"]


# ---------------------------------------------------------------------------
# recommendation_facts
# ---------------------------------------------------------------------------

class TestRecommendationFacts:
    def _inputs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        summary = pd.concat([
            pd.DataFrame({
                "experiment": ["no_aug"] * 3,
                "batch_size": [8, 16, 32],
                "median": [200.0, 190.0, 180.0],
                "p10": [190.0] * 3, "p90": [210.0] * 3,
                "median_time_per_step_ms": [40.0, 80.0, 160.0],
                "eff_max_pct": [100.0, 95.0, 90.0],
            }),
            pd.DataFrame({
                "experiment": ["with_aug"] * 3,
                "batch_size": [8, 16, 32],
                "median": [195.0, 185.0, 175.0],
                "p10": [185.0] * 3, "p90": [205.0] * 3,
                "median_time_per_step_ms": [42.0, 82.0, 162.0],
                "eff_max_pct": [100.0, 94.9, 89.7],
            }),
        ], ignore_index=True)
        gpu_tbl = pd.DataFrame({
            "experiment": ["no_aug", "no_aug", "no_aug",
                           "with_aug", "with_aug", "with_aug"],
            "batch_size": [8, 16, 32, 8, 16, 32],
            "median_gpu_util_pct": [70.0, 82.0, 90.0, 71.0, 83.0, 91.0],
            "median_gpu_mem_gb": [0.4, 0.7, 1.3, 0.4, 0.7, 1.3],
            "peak_gpu_mem_gb": [0.5, 0.8, 1.5, 0.5, 0.8, 1.5],
        })
        return summary, gpu_tbl

    def test_recommended_bs_stored(self) -> None:
        s, g = self._inputs()
        facts = recommendation_facts(s, g, recommended_bs=16)
        assert facts["recommended_bs"] == 16

    def test_plateau_bs_list(self) -> None:
        s, g = self._inputs()
        facts = recommendation_facts(s, g, recommended_bs=16, threshold_pct=90.0)
        assert isinstance(facts["plateau_bs"], list)

    def test_per_exp_keys_present(self) -> None:
        s, g = self._inputs()
        facts = recommendation_facts(s, g, recommended_bs=16)
        assert "no_aug" in facts["per_exp"]
        assert "with_aug" in facts["per_exp"]

    def test_gpu_mem_dict_keyed_by_batch_size(self) -> None:
        s, g = self._inputs()
        facts = recommendation_facts(s, g, recommended_bs=16)
        assert 16 in facts["gpu_mem"]

    def test_aug_penalty_present_when_both_experiments(self) -> None:
        s, g = self._inputs()
        facts = recommendation_facts(s, g, recommended_bs=16)
        assert "aug_penalty_pct_at_recommended" in facts
