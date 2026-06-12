"""Unit tests for SkiNet.Utils.analysis.aggregation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from unittest.mock import patch

from SkiNet.Utils.analysis.aggregation import (
    _best_metric_columns,
    _latest_value,
    epoch_metrics,
    load_runs,
    metric_inventory,
    parameter_inventory,
    rank_runs,
    summarize_by_family,
    summarize_runs,
)
from SkiNet.Utils.analysis.lr_sweep import (
    arch_consistency,
    best_run_per_group,
    load_sweep_runs,
    pivot_dim_effect,
    rank_all_runs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RUN_A = "aabbccdd-0000-0000-0000-000000000001"
RUN_B = "bbccddee-0000-0000-0000-000000000002"

# Experiment names that satisfy arch_pattern: enc-<name>_merge-<name>
EXP_A = "enc-resnet50_merge-add"
EXP_B = "enc-efficientnet_merge-concat"

# Timestamps in ms; duration_min = (end - start) / 60_000
START_A, END_A = 0, 120_000      # 2 min
START_B, END_B = 0, 60_000       # 1 min


def _runs_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_uuid": [RUN_A, RUN_B],
            "experiment_id": [1, 1],
            "experiment_name": [EXP_A, EXP_B],
            "status": ["FINISHED", "FINISHED"],
            "start_time": [START_A, START_B],
            "end_time": [END_A, END_B],
        }
    )


def _latest_df() -> pd.DataFrame:
    rows = []
    for run, train, val, best, iou, loss, sps, tps, grad in [
        (RUN_A, 0.85, 0.80, 0.81, 0.72, 0.30, 12.0, 83.0, 128.0),
        (RUN_B, 0.78, 0.76, 0.77, 0.68, 0.35, 11.0, 91.0, 64.0),
    ]:
        for key, val_ in [
            ("final/train_dice", train),
            ("final/val_dice", val),
            ("final/val_best_dice_at_threshold", best),
            ("final/val_iou", iou),
            ("final/val_loss", loss),
            ("final/perf/samples_per_sec", sps),
            ("final/perf/time_per_step_ms", tps),
            ("final/grad_scale", grad),
        ]:
            rows.append({"run_uuid": run, "key": key, "value": val_})
    return pd.DataFrame(rows)


def _metrics_df() -> pd.DataFrame:
    """Three epochs of val_dice / val_iou / val_loss / val_best_dice_at_threshold per run."""
    rows = []
    for run, dice_vals, iou_vals, loss_vals, best_vals in [
        (RUN_A, [0.70, 0.80, 0.75], [0.60, 0.72, 0.68], [0.50, 0.35, 0.40], [0.71, 0.81, 0.76]),
        (RUN_B, [0.65, 0.76, 0.72], [0.55, 0.68, 0.65], [0.55, 0.38, 0.42], [0.66, 0.77, 0.73]),
    ]:
        for step, (d, i, l, b) in enumerate(zip(dice_vals, iou_vals, loss_vals, best_vals), start=1):
            for key, v in [
                ("val_dice", d),
                ("val_iou", i),
                ("val_loss", l),
                ("val_best_dice_at_threshold", b),
            ]:
                rows.append({"run_uuid": run, "key": key, "value": v, "step": step, "timestamp": step})
    return pd.DataFrame(rows)


@pytest.fixture()
def tables() -> dict[str, pd.DataFrame]:
    return {"runs": _runs_df(), "metrics": _metrics_df(), "latest": _latest_df()}


# ---------------------------------------------------------------------------
# _latest_value
# ---------------------------------------------------------------------------

class TestLatestValue:
    def test_returns_value_when_key_exists(self) -> None:
        df = pd.DataFrame({"key": ["a", "b"], "value": [1.5, 2.5]})
        assert _latest_value(df, "a") == pytest.approx(1.5)

    def test_returns_nan_when_key_missing(self) -> None:
        df = pd.DataFrame({"key": ["a"], "value": [1.0]})
        assert math.isnan(_latest_value(df, "missing"))


# ---------------------------------------------------------------------------
# _best_metric_columns
# ---------------------------------------------------------------------------

class TestBestMetricColumns:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"key": ["val_dice"] * 3, "value": [0.7, 0.9, 0.8], "step": [1, 2, 3], "timestamp": [1, 2, 3]}
        )

    def test_max_mode_picks_highest_value(self) -> None:
        result = _best_metric_columns(self._df(), "val_dice", "max")
        assert result["val_dice_max"] == pytest.approx(0.9)
        assert result["val_dice_max_epoch"] == 2

    def test_min_mode_picks_lowest_value(self) -> None:
        result = _best_metric_columns(self._df(), "val_dice", "min")
        assert result["val_dice_min"] == pytest.approx(0.7)
        assert result["val_dice_min_epoch"] == 1

    def test_empty_returns_nan(self) -> None:
        empty = pd.DataFrame(columns=["key", "value", "step", "timestamp"])
        result = _best_metric_columns(empty, "val_dice", "max")
        assert math.isnan(result["val_dice_max"])
        assert math.isnan(result["val_dice_max_epoch"])

    def test_step_preserved(self) -> None:
        result = _best_metric_columns(self._df(), "val_dice", "max")
        assert result["val_dice_max_step"] == 2

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="median.*val_dice|val_dice.*median"):
            _best_metric_columns(self._df(), "val_dice", "median")


# ---------------------------------------------------------------------------
# summarize_runs
# ---------------------------------------------------------------------------

class TestSummarizeRuns:
    def test_one_row_per_run(self, tables: dict[str, pd.DataFrame]) -> None:
        result = summarize_runs(tables, monitor="val_dice")
        assert len(result) == 2

    def test_sorted_by_monitor_descending(self, tables: dict[str, pd.DataFrame]) -> None:
        result = summarize_runs(tables, monitor="val_dice")
        assert result.iloc[0]["val_dice_max"] >= result.iloc[1]["val_dice_max"]

    def test_custom_monitor_column_present(self, tables: dict[str, pd.DataFrame]) -> None:
        result = summarize_runs(tables, monitor="val_dice")
        assert "val_dice_max" in result.columns

    def test_encoder_merge_parsed(self, tables: dict[str, pd.DataFrame]) -> None:
        result = summarize_runs(tables, monitor="val_dice")
        row_a = result[result["run_uuid"] == RUN_A].iloc[0]
        assert row_a["encoder"] == "resnet50"
        assert row_a["merge"] == "add"

    def test_duration_min_correct(self, tables: dict[str, pd.DataFrame]) -> None:
        result = summarize_runs(tables, monitor="val_dice")
        row_a = result[result["run_uuid"] == RUN_A].iloc[0]
        assert row_a["duration_min"] == pytest.approx(2.0)

    def test_run_short_is_8_chars(self, tables: dict[str, pd.DataFrame]) -> None:
        result = summarize_runs(tables, monitor="val_dice")
        assert all(result["run_short"].str.len() == 8)

    def test_final_metrics_populated(self, tables: dict[str, pd.DataFrame]) -> None:
        result = summarize_runs(tables, monitor="val_dice")
        row_a = result[result["run_uuid"] == RUN_A].iloc[0]
        assert row_a["final_train_dice"] == pytest.approx(0.85)
        assert row_a["final_val_dice"] == pytest.approx(0.80)
        assert row_a["samples_per_sec"] == pytest.approx(12.0)

    def test_generalization_gap(self, tables: dict[str, pd.DataFrame]) -> None:
        result = summarize_runs(tables, monitor="val_dice")
        row_a = result[result["run_uuid"] == RUN_A].iloc[0]
        expected_gap = pytest.approx(0.85 - 0.80)
        assert row_a["generalization_gap_final"] == expected_gap

    def test_best_val_dice_epoch_correct(self, tables: dict[str, pd.DataFrame]) -> None:
        # RUN_A val_dice: [0.70, 0.80, 0.75] → best at epoch 2
        result = summarize_runs(tables, monitor="val_dice")
        row_a = result[result["run_uuid"] == RUN_A].iloc[0]
        assert row_a["val_dice_max_epoch"] == 2

    def test_min_val_loss_epoch_correct(self, tables: dict[str, pd.DataFrame]) -> None:
        # RUN_A val_loss: [0.50, 0.35, 0.40] → min at epoch 2
        result = summarize_runs(tables, monitor="val_dice")
        row_a = result[result["run_uuid"] == RUN_A].iloc[0]
        assert row_a["val_loss_min_epoch"] == 2

    def test_missing_latest_key_gives_nan(self, tables: dict[str, pd.DataFrame]) -> None:
        # Remove all final/val_iou entries
        tables["latest"] = tables["latest"][tables["latest"]["key"] != "final/val_iou"]
        result = summarize_runs(tables, monitor="val_dice")
        assert result["final_val_iou"].isna().all()

    def test_empty_runs_returns_empty_dataframe(self) -> None:
        tables = {
            "runs": pd.DataFrame(columns=["run_uuid", "experiment_id", "experiment_name", "status", "start_time", "end_time"]),
            "metrics": pd.DataFrame(columns=["run_uuid", "key", "value", "step", "timestamp"]),
            "latest": pd.DataFrame(columns=["run_uuid", "key", "value"]),
        }
        result = summarize_runs(tables, monitor="val_dice")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# metric_inventory
# ---------------------------------------------------------------------------

class TestMetricInventory:
    def test_one_row_per_key(self) -> None:
        metrics = pd.DataFrame(
            {
                "run_uuid": ["r1", "r1", "r2"],
                "key": ["val_dice", "val_dice", "val_loss"],
                "value": [0.8, 0.9, 0.3],
                "step": [1, 2, 1],
            }
        )
        result = metric_inventory(metrics)
        assert set(result["key"]) == {"val_dice", "val_loss"}

    def test_run_coverage_counted(self) -> None:
        metrics = pd.DataFrame(
            {
                "run_uuid": ["r1", "r2", "r2"],
                "key": ["val_dice"] * 3,
                "value": [0.8, 0.7, 0.9],
                "step": [1, 1, 2],
            }
        )
        result = metric_inventory(metrics)
        assert result.iloc[0]["runs"] == 2

    def test_sorted_by_key(self) -> None:
        metrics = pd.DataFrame(
            {"run_uuid": ["r1", "r1"], "key": ["z_metric", "a_metric"], "value": [1.0, 2.0], "step": [1, 1]}
        )
        result = metric_inventory(metrics)
        assert list(result["key"]) == ["a_metric", "z_metric"]


# ---------------------------------------------------------------------------
# parameter_inventory
# ---------------------------------------------------------------------------

class TestParameterInventory:
    def _params(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "run_uuid": ["r1", "r1", "r2", "r2"],
                "key": ["lr", "batch_size", "lr", "batch_size"],
                "value": ["0.001", "8", "0.01", "8"],
            }
        )

    def test_constant_params_split(self) -> None:
        const, variable = parameter_inventory(self._params())
        assert "batch_size" in const["param"].values

    def test_variable_params_split(self) -> None:
        const, variable = parameter_inventory(self._params())
        assert "lr" in variable["param"].values

    def test_values_truncated_at_4(self) -> None:
        params = pd.DataFrame(
            {
                "run_uuid": [f"r{i}" for i in range(6)],
                "key": ["lr"] * 6,
                "value": [str(v) for v in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]],
            }
        )
        _, variable = parameter_inventory(params)
        row = variable[variable["param"] == "lr"].iloc[0]
        assert row["values"].endswith("...")


# ---------------------------------------------------------------------------
# summarize_by_family
# ---------------------------------------------------------------------------

class TestFamilySummary:
    def _run_summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "run_uuid": [RUN_A, RUN_B],
                "encoder": ["resnet50", "efficientnet"],
                "merge": ["add", "concat"],
                "val_dice_max": [0.81, 0.77],
                "val_dice_tail_mean": [0.79, 0.75],
                "final_val_dice": [0.80, 0.76],
                "samples_per_sec": [12.0, 11.0],
                "generalization_gap_final": [0.05, 0.02],
            }
        )

    def test_contains_encoder_and_merge_families(self) -> None:
        result = summarize_by_family(self._run_summary(), monitor="val_dice").data
        assert set(result["family"]) == {"encoder", "merge"}

    def test_sorted_within_family_descending_dice(self) -> None:
        summary = self._run_summary()
        summary = pd.concat([summary, summary.assign(
            run_uuid=["rx", "ry"], encoder=["resnet50", "resnet50"],
            val_dice_max=[0.90, 0.70]
        )], ignore_index=True)
        result = summarize_by_family(summary, monitor="val_dice", group_cols=["encoder"]).data
        encoder_rows = result[result["family"] == "encoder"].reset_index(drop=True)
        assert encoder_rows.iloc[0]["mean_best_dice"] >= encoder_rows.iloc[1]["mean_best_dice"]

    def test_n_counts_runs_per_group(self) -> None:
        result = summarize_by_family(self._run_summary(), monitor="val_dice", group_cols=["encoder"]).data
        assert result["n"].sum() == 2


# ---------------------------------------------------------------------------
# epoch_metrics
# ---------------------------------------------------------------------------

class TestEpochMetrics:
    def test_epoch_numbering_starts_at_1(self, tables: dict[str, pd.DataFrame]) -> None:
        run_summary = pd.DataFrame(
            {"run_uuid": [RUN_A, RUN_B], "encoder": ["resnet50", "efficientnet"], "merge": ["add", "concat"]}
        )
        result = epoch_metrics(tables["metrics"], run_summary, keys=["val_dice"])
        epochs = result[result["run_uuid"] == RUN_A]["epoch"].tolist()
        assert epochs == [1, 2, 3]

    def test_architecture_column_format(self, tables: dict[str, pd.DataFrame]) -> None:
        run_summary = pd.DataFrame(
            {"run_uuid": [RUN_A], "encoder": ["resnet50"], "merge": ["add"]}
        )
        result = epoch_metrics(tables["metrics"], run_summary, keys=["val_dice"])
        assert result["architecture"].iloc[0] == "resnet50 + add"

    def test_filters_to_requested_keys(self, tables: dict[str, pd.DataFrame]) -> None:
        run_summary = pd.DataFrame(
            {"run_uuid": [RUN_A, RUN_B], "encoder": ["resnet50", "efficientnet"], "merge": ["add", "concat"]}
        )
        result = epoch_metrics(tables["metrics"], run_summary, keys=["val_dice"])
        assert set(result["key"]) == {"val_dice"}


# ---------------------------------------------------------------------------
# rank_runs
# ---------------------------------------------------------------------------

class TestRankTable:
    def _run_summary(self) -> pd.DataFrame:
        # RUN_B has the higher peak (0.85); RUN_A has the higher tail mean (0.80)
        return pd.DataFrame({
            "run_uuid": [RUN_A, RUN_B],
            "run_short": [RUN_A[:8], RUN_B[:8]],
            "encoder": ["resnet50", "efficientnet"],
            "merge": ["add", "concat"],
            "status": ["FINISHED", "FINISHED"],
            "val_dice_max": [0.81, 0.85],
            "val_dice_tail_mean": [0.80, 0.77],
            "val_dice_max_epoch": [2, 3],
            "val_iou_max": [0.72, 0.75],
            "val_loss_min": [0.30, 0.28],
            "final_val_dice": [0.79, 0.82],
            "generalization_gap_final": [0.05, 0.04],
            "samples_per_sec": [12.0, 11.0],
            "duration_min": [2.0, 1.0],
        })

    def test_returns_styler(self) -> None:
        from pandas.io.formats.style import Styler
        assert isinstance(rank_runs(self._run_summary(), monitor="val_dice"), Styler)

    def test_sort_by_best_puts_highest_peak_first(self) -> None:
        # RUN_B peak=0.85 > RUN_A peak=0.81
        df = rank_runs(self._run_summary(), monitor="val_dice", sort_by="best").data
        assert df.iloc[0]["val_dice_max"] == pytest.approx(0.85)

    def test_sort_by_tail_puts_highest_tail_mean_first(self) -> None:
        # RUN_A tail_mean=0.80 > RUN_B tail_mean=0.77
        df = rank_runs(self._run_summary(), monitor="val_dice", sort_by="tail").data
        assert df.iloc[0]["val_dice_tail_mean"] == pytest.approx(0.80)

    def test_invalid_sort_by_raises(self) -> None:
        with pytest.raises(ValueError, match="sort_by"):
            rank_runs(self._run_summary(), monitor="val_dice", sort_by="median")

    def test_caption_best_names_sort_col_and_criterion(self) -> None:
        caption = rank_runs(self._run_summary(), monitor="val_dice", sort_by="best").caption
        assert "val_dice_max" in caption
        assert "peak value" in caption

    def test_caption_tail_names_sort_col_and_window(self) -> None:
        caption = rank_runs(self._run_summary(), monitor="val_dice", sort_by="tail", tail_n=5).caption
        assert "val_dice_tail_mean" in caption
        assert "5" in caption


# ---------------------------------------------------------------------------
# load_runs
# ---------------------------------------------------------------------------

def _mock_tables(run_uuids: list, run_names: list, exp_ids: list) -> dict:
    return {"runs": pd.DataFrame({
        "run_uuid": run_uuids,
        "run_name": run_names,
        "experiment_id": exp_ids,
    })}


def _mock_summary(run_uuids: list) -> pd.DataFrame:
    return pd.DataFrame({
        "run_uuid": run_uuids,
        "val_dice_max": [0.80] * len(run_uuids),
    })


class TestLoadRuns:
    EXP_MAP = {1: "arch_a", 2: "arch_b"}

    @patch("SkiNet.Utils.analysis.aggregation.summarize_runs")
    @patch("SkiNet.Utils.analysis.aggregation.load_mlflow_tables")
    def test_returns_dataframe(self, mock_load: Any, mock_summarize: Any, tmp_path: Path) -> None:
        uuids = ["a1", "a2", "b1", "b2"]
        names = ["run_seed100", "run_seed101", "run_seed100", "run_seed101"]
        exp_ids = [1, 1, 2, 2]
        mock_load.return_value = _mock_tables(uuids, names, exp_ids)
        mock_summarize.return_value = _mock_summary(uuids)
        df = load_runs(tmp_path / "a.db", exp_map=self.EXP_MAP, monitor="val_dice")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4

    @patch("SkiNet.Utils.analysis.aggregation.summarize_runs")
    @patch("SkiNet.Utils.analysis.aggregation.load_mlflow_tables")
    def test_seed_extracted_correctly(self, mock_load: Any, mock_summarize: Any, tmp_path: Path) -> None:
        uuids = ["a1", "a2", "b1", "b2"]
        names = ["run_seed100", "run_seed101", "run_seed100", "run_seed101"]
        exp_ids = [1, 1, 2, 2]
        mock_load.return_value = _mock_tables(uuids, names, exp_ids)
        mock_summarize.return_value = _mock_summary(uuids)
        df = load_runs(tmp_path / "a.db", exp_map=self.EXP_MAP, monitor="val_dice")
        assert sorted(df["seed"].unique()) == [100, 101]

    @patch("SkiNet.Utils.analysis.aggregation.summarize_runs")
    @patch("SkiNet.Utils.analysis.aggregation.load_mlflow_tables")
    def test_arch_mapped_from_exp_id(self, mock_load: Any, mock_summarize: Any, tmp_path: Path) -> None:
        uuids = ["a1", "b1"]
        names = ["run_seed100", "run_seed100"]
        exp_ids = [1, 2]
        mock_load.return_value = _mock_tables(uuids, names, exp_ids)
        mock_summarize.return_value = _mock_summary(uuids)
        df = load_runs(tmp_path / "a.db", exp_map=self.EXP_MAP, monitor="val_dice")
        assert set(df["arch"]) == {"arch_a", "arch_b"}

    @patch("SkiNet.Utils.analysis.aggregation.summarize_runs")
    @patch("SkiNet.Utils.analysis.aggregation.load_mlflow_tables")
    def test_two_dbs_concatenated(self, mock_load: Any, mock_summarize: Any, tmp_path: Path) -> None:
        uuids_p1 = ["a1", "b1"]
        uuids_p2 = ["a2", "b2"]
        names_p1 = ["run_seed100", "run_seed100"]
        names_p2 = ["run_seed101", "run_seed101"]
        exp_ids = [1, 2]
        mock_load.side_effect = [
            _mock_tables(uuids_p1, names_p1, exp_ids),
            _mock_tables(uuids_p2, names_p2, exp_ids),
        ]
        mock_summarize.side_effect = [
            _mock_summary(uuids_p1),
            _mock_summary(uuids_p2),
        ]
        df = load_runs(tmp_path / "a.db", tmp_path / "b.db", exp_map=self.EXP_MAP, monitor="val_dice")
        assert len(df) == 4

    @patch("SkiNet.Utils.analysis.aggregation.summarize_runs")
    @patch("SkiNet.Utils.analysis.aggregation.load_mlflow_tables")
    def test_unbalanced_raises(self, mock_load: Any, mock_summarize: Any, tmp_path: Path) -> None:
        uuids = ["a1", "a2", "b1"]
        names = ["run_seed100", "run_seed101", "run_seed100"]
        exp_ids = [1, 1, 2]
        mock_load.return_value = _mock_tables(uuids, names, exp_ids)
        mock_summarize.return_value = _mock_summary(uuids)
        with pytest.raises(ValueError, match=r"Unbalanced seed coverage.*differ.*arch_b"):
            load_runs(tmp_path / "a.db", exp_map=self.EXP_MAP, monitor="val_dice")


# ---------------------------------------------------------------------------
# Cross-LR sweep fixture
# ---------------------------------------------------------------------------

def _sweep_df() -> pd.DataFrame:
    """A small 2-LR × 2-encoder × 2-merge sweep (8 runs) for the cross-LR helpers.

    Values are chosen so that within each LR a known architecture wins, and so
    the two LRs are cleanly separated on every metric — letting tests assert on
    ordering and selection without depending on float ties.
    """
    rows = []
    # (lr, encoder, merge, tail_mean, dice_max, tail_std, max_epoch, gap, drop, sps)
    spec = [
        ("3e-4", "classical", "he2", 0.826, 0.843, 0.011, 97, 0.078, 0.017, 140.0),
        ("3e-4", "classical", "classical", 0.818, 0.835, 0.010, 90, 0.082, 0.018, 150.0),
        ("3e-4", "se", "he2", 0.809, 0.825, 0.009, 70, 0.085, 0.020, 130.0),
        ("3e-4", "se", "classical", 0.804, 0.820, 0.009, 65, 0.088, 0.021, 138.0),
        ("1e-4", "classical", "he2", 0.819, 0.827, 0.006, 95, 0.070, 0.012, 128.0),
        ("1e-4", "classical", "classical", 0.812, 0.821, 0.006, 88, 0.073, 0.013, 145.0),
        ("1e-4", "se", "he2", 0.802, 0.812, 0.005, 60, 0.075, 0.014, 126.0),
        ("1e-4", "se", "classical", 0.798, 0.808, 0.005, 55, 0.078, 0.015, 133.0),
    ]
    for i, (lr, enc, mrg, tm, dm, ts, me, gap, drop, sps) in enumerate(spec):
        rows.append({
            "run_uuid": f"run-{i:02d}",
            "encoder": enc,
            "merge": mrg,
            "lr": lr,
            "val_dice_tail_mean": tm,
            "val_dice_max": dm,
            "val_dice_tail_std": ts,
            "val_dice_max_epoch": me,
            "generalization_gap_final": gap,
            "drop_peak_to_final": drop,
            "samples_per_sec": sps,
        })
    return pd.DataFrame(rows)


@pytest.fixture()
def sweep() -> pd.DataFrame:
    return _sweep_df()


# ---------------------------------------------------------------------------
# load_sweep_runs
# ---------------------------------------------------------------------------

class TestLoadSweepRuns:
    EXPERIMENTS = {
        "3e-4": ("lr_3e-4.db", "exp-3e-4"),
        "1e-4": ("lr_1e-4.db", "exp-1e-4"),
    }

    @patch("SkiNet.Utils.analysis.lr_sweep.summarize_runs")
    @patch("SkiNet.Utils.analysis.lr_sweep.load_mlflow_tables")
    def test_concatenates_all_experiments(self, mock_load: Any, mock_summarize: Any, tmp_path: Path) -> None:
        mock_load.return_value = {"runs": pd.DataFrame()}
        mock_summarize.side_effect = [
            pd.DataFrame({"run_uuid": ["a", "b"]}),
            pd.DataFrame({"run_uuid": ["c", "d"]}),
        ]
        df = load_sweep_runs(self.EXPERIMENTS, tmp_path, monitor="val_dice")
        assert len(df) == 4

    @patch("SkiNet.Utils.analysis.lr_sweep.summarize_runs")
    @patch("SkiNet.Utils.analysis.lr_sweep.load_mlflow_tables")
    def test_group_label_tagged_per_experiment(self, mock_load: Any, mock_summarize: Any, tmp_path: Path) -> None:
        mock_load.return_value = {"runs": pd.DataFrame()}
        mock_summarize.side_effect = [
            pd.DataFrame({"run_uuid": ["a", "b"]}),
            pd.DataFrame({"run_uuid": ["c"]}),
        ]
        df = load_sweep_runs(self.EXPERIMENTS, tmp_path, monitor="val_dice")
        assert df.loc[df["run_uuid"] == "a", "lr"].iloc[0] == "3e-4"
        assert df.loc[df["run_uuid"] == "c", "lr"].iloc[0] == "1e-4"

    @patch("SkiNet.Utils.analysis.lr_sweep.summarize_runs")
    @patch("SkiNet.Utils.analysis.lr_sweep.load_mlflow_tables")
    def test_group_by_column_name_is_configurable(self, mock_load: Any, mock_summarize: Any, tmp_path: Path) -> None:
        mock_load.return_value = {"runs": pd.DataFrame()}
        mock_summarize.side_effect = [pd.DataFrame({"run_uuid": ["a"]}), pd.DataFrame({"run_uuid": ["b"]})]
        df = load_sweep_runs(self.EXPERIMENTS, tmp_path, monitor="val_dice", group_by="phase")
        assert "phase" in df.columns
        assert "lr" not in df.columns

    @patch("SkiNet.Utils.analysis.lr_sweep.summarize_runs")
    @patch("SkiNet.Utils.analysis.lr_sweep.load_mlflow_tables")
    def test_resolves_db_path_relative_to_mlruns(self, mock_load: Any, mock_summarize: Any, tmp_path: Path) -> None:
        mock_load.return_value = {"runs": pd.DataFrame()}
        mock_summarize.return_value = pd.DataFrame({"run_uuid": ["a"]})
        load_sweep_runs({"3e-4": ("lr_3e-4.db", "exp")}, tmp_path, monitor="val_dice")
        called_path = mock_load.call_args[0][0]
        assert called_path == tmp_path / "lr_3e-4.db"

    @patch("SkiNet.Utils.analysis.lr_sweep.summarize_runs")
    @patch("SkiNet.Utils.analysis.lr_sweep.load_mlflow_tables")
    def test_index_is_reset_contiguous(self, mock_load: Any, mock_summarize: Any, tmp_path: Path) -> None:
        mock_load.return_value = {"runs": pd.DataFrame()}
        mock_summarize.side_effect = [
            pd.DataFrame({"run_uuid": ["a", "b"]}),
            pd.DataFrame({"run_uuid": ["c", "d"]}),
        ]
        df = load_sweep_runs(self.EXPERIMENTS, tmp_path, monitor="val_dice")
        assert df.index.tolist() == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# best_run_per_group
# ---------------------------------------------------------------------------

class TestBestPerGroup:
    def test_returns_styler(self, sweep: pd.DataFrame) -> None:
        from pandas.io.formats.style import Styler
        assert isinstance(best_run_per_group(sweep), Styler)

    def test_one_row_per_group(self, sweep: pd.DataFrame) -> None:
        result = best_run_per_group(sweep).data
        assert len(result) == sweep["lr"].nunique()

    def test_picks_highest_tail_mean_within_each_group(self, sweep: pd.DataFrame) -> None:
        result = best_run_per_group(sweep).data
        # classical+he2 is the per-LR tail-mean winner at both LRs
        for lr in sweep["lr"].unique():
            row = result[result["lr"] == lr].iloc[0]
            assert (row["encoder"], row["merge"]) == ("classical", "he2")

    def test_sorted_by_rank_metric_descending(self, sweep: pd.DataFrame) -> None:
        result = best_run_per_group(sweep).data
        tail = result["val_dice_tail_mean"].tolist()
        assert tail == sorted(tail, reverse=True)

    def test_custom_rank_metric(self, sweep: pd.DataFrame) -> None:
        # Ranking by throughput selects classical+classical (the fastest) at each LR
        result = best_run_per_group(sweep, rank_by="samples_per_sec").data
        for lr in sweep["lr"].unique():
            row = result[result["lr"] == lr].iloc[0]
            assert row["merge"] == "classical"


# ---------------------------------------------------------------------------
# rank_all_runs
# ---------------------------------------------------------------------------

class TestRankAllRuns:
    def test_returns_styler(self, sweep: pd.DataFrame) -> None:
        from pandas.io.formats.style import Styler
        assert isinstance(rank_all_runs(sweep), Styler)

    def test_keeps_every_run(self, sweep: pd.DataFrame) -> None:
        result = rank_all_runs(sweep).data
        assert len(result) == len(sweep)

    def test_sorted_by_tail_mean_descending(self, sweep: pd.DataFrame) -> None:
        result = rank_all_runs(sweep).data
        tail = result["val_dice_tail_mean"].tolist()
        assert tail == sorted(tail, reverse=True)

    def test_top_run_is_overall_best_tail_mean(self, sweep: pd.DataFrame) -> None:
        result = rank_all_runs(sweep).data
        top = result.iloc[0]
        assert (top["lr"], top["encoder"], top["merge"]) == ("3e-4", "classical", "he2")

    def test_columns_renamed_to_compact_aliases(self, sweep: pd.DataFrame) -> None:
        result = rank_all_runs(sweep).data
        assert {"tail_std", "epoch", "gen_gap", "drop"} <= set(result.columns)
        assert "val_dice_tail_std" not in result.columns

    def test_sort_by_peak_reorders(self, sweep: pd.DataFrame) -> None:
        result = rank_all_runs(sweep, sort_by="val_dice_max").data
        peaks = result["val_dice_max"].tolist()
        assert peaks == sorted(peaks, reverse=True)

    def test_missing_optional_column_tolerated(self, sweep: pd.DataFrame) -> None:
        trimmed = sweep.drop(columns=["drop_peak_to_final"])
        result = rank_all_runs(trimmed).data
        assert "drop" not in result.columns
        assert len(result) == len(trimmed)


# ---------------------------------------------------------------------------
# pivot_dim_effect
# ---------------------------------------------------------------------------

class TestPivotDimEffect:
    def test_returns_styler(self, sweep: pd.DataFrame) -> None:
        from pandas.io.formats.style import Styler
        assert isinstance(pivot_dim_effect(sweep, "encoder"), Styler)

    def test_rows_follow_group_order(self, sweep: pd.DataFrame) -> None:
        result = pivot_dim_effect(sweep, "encoder", group_order=["1e-4", "3e-4"]).data
        assert result.index.tolist() == ["1e-4", "3e-4"]

    def test_outer_columns_follow_dim_order(self, sweep: pd.DataFrame) -> None:
        result = pivot_dim_effect(sweep, "encoder", dim_order=["classical", "se"]).data
        outer = list(dict.fromkeys(c[0] for c in result.columns))
        assert outer == ["classical", "se"]

    def test_mean_and_std_columns_present_per_dim(self, sweep: pd.DataFrame) -> None:
        result = pivot_dim_effect(sweep, "encoder", dim_order=["classical", "se"]).data
        assert ("classical", "mean") in result.columns
        assert ("classical", "std") in result.columns

    def test_mean_matches_groupby_mean(self, sweep: pd.DataFrame) -> None:
        result = pivot_dim_effect(sweep, "encoder").data
        expected = sweep[(sweep.lr == "3e-4") & (sweep.encoder == "classical")]["val_dice_tail_mean"].mean()
        assert result.loc["3e-4", ("classical", "mean")] == pytest.approx(expected)

    def test_absent_dim_value_becomes_nan_column(self, sweep: pd.DataFrame) -> None:
        result = pivot_dim_effect(sweep, "encoder", dim_order=["classical", "he2"]).data
        # he2 is not an encoder in this fixture -> NaN column
        assert result[("he2", "mean")].isna().all()


# ---------------------------------------------------------------------------
# arch_consistency
# ---------------------------------------------------------------------------

class TestArchConsistency:
    def test_returns_styler(self, sweep: pd.DataFrame) -> None:
        from pandas.io.formats.style import Styler
        assert isinstance(arch_consistency(sweep), Styler)

    def test_one_row_per_architecture(self, sweep: pd.DataFrame) -> None:
        result = arch_consistency(sweep).data
        assert len(result) == 4  # 2 encoders × 2 merges

    def test_sorted_by_mean_descending(self, sweep: pd.DataFrame) -> None:
        result = arch_consistency(sweep).data
        means = result["mean_tail_mean"].tolist()
        assert means == sorted(means, reverse=True)

    def test_top_architecture_has_highest_cross_lr_mean(self, sweep: pd.DataFrame) -> None:
        result = arch_consistency(sweep).data
        top = result.iloc[0]
        assert (top["encoder"], top["merge"]) == ("classical", "he2")

    def test_mean_std_min_max_match_manual(self, sweep: pd.DataFrame) -> None:
        result = arch_consistency(sweep).data
        row = result[(result["encoder"] == "classical") & (result["merge"] == "he2")].iloc[0]
        vals = sweep[(sweep["encoder"] == "classical") & (sweep["merge"] == "he2")]["val_dice_tail_mean"]
        assert row["mean_tail_mean"] == pytest.approx(vals.mean())
        assert row["std_tail_mean"] == pytest.approx(vals.std())
        assert row["min_tail_mean"] == pytest.approx(vals.min())
        assert row["max_tail_mean"] == pytest.approx(vals.max())

    def test_value_col_drives_column_suffix(self, sweep: pd.DataFrame) -> None:
        result = arch_consistency(sweep, value_col="val_dice_max").data
        assert {"mean_max", "std_max", "min_max", "max_max"} <= set(result.columns)
