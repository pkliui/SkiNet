"""Unit tests for SkiNet.Utils.analysis.aggregation."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from SkiNet.Utils.analysis.aggregation import (
    _best_metric_columns,
    _latest_value,
    epoch_metrics,
    family_summary,
    metric_inventory,
    parameter_inventory,
    rank_table,
    summarize_runs,
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
        result = _best_metric_columns(self._df(), "val_dice", "best_val_dice", "max")
        assert result["best_val_dice"] == pytest.approx(0.9)
        assert result["best_val_dice_epoch"] == 2

    def test_min_mode_picks_lowest_value(self) -> None:
        result = _best_metric_columns(self._df(), "val_dice", "min_val_dice", "min")
        assert result["min_val_dice"] == pytest.approx(0.7)
        assert result["min_val_dice_epoch"] == 1

    def test_empty_returns_nan(self) -> None:
        empty = pd.DataFrame(columns=["key", "value", "step", "timestamp"])
        result = _best_metric_columns(empty, "val_dice", "best_val_dice", "max")
        assert math.isnan(result["best_val_dice"])
        assert math.isnan(result["best_val_dice_epoch"])

    def test_step_preserved(self) -> None:
        result = _best_metric_columns(self._df(), "val_dice", "best_val_dice", "max")
        assert result["best_val_dice_step"] == 2


# ---------------------------------------------------------------------------
# summarize_runs
# ---------------------------------------------------------------------------

class TestSummarizeRuns:
    def test_one_row_per_run(self, tables: dict[str, pd.DataFrame]) -> None:
        result = summarize_runs(tables)
        assert len(result) == 2

    def test_sorted_by_monitor_descending(self, tables: dict[str, pd.DataFrame]) -> None:
        result = summarize_runs(tables)
        assert result.iloc[0]["best_val_best_dice_at_threshold"] >= result.iloc[1]["best_val_best_dice_at_threshold"]

    def test_custom_monitor_column_present(self, tables: dict[str, pd.DataFrame]) -> None:
        result = summarize_runs(tables, monitor="val_dice")
        assert "best_val_dice" in result.columns

    def test_encoder_merge_parsed(self, tables: dict[str, pd.DataFrame]) -> None:
        result = summarize_runs(tables)
        row_a = result[result["run_uuid"] == RUN_A].iloc[0]
        assert row_a["encoder"] == "resnet50"
        assert row_a["merge"] == "add"

    def test_duration_min_correct(self, tables: dict[str, pd.DataFrame]) -> None:
        result = summarize_runs(tables)
        row_a = result[result["run_uuid"] == RUN_A].iloc[0]
        assert row_a["duration_min"] == pytest.approx(2.0)

    def test_run_short_is_8_chars(self, tables: dict[str, pd.DataFrame]) -> None:
        result = summarize_runs(tables)
        assert all(result["run_short"].str.len() == 8)

    def test_final_metrics_populated(self, tables: dict[str, pd.DataFrame]) -> None:
        result = summarize_runs(tables)
        row_a = result[result["run_uuid"] == RUN_A].iloc[0]
        assert row_a["final_train_dice"] == pytest.approx(0.85)
        assert row_a["final_val_dice"] == pytest.approx(0.80)
        assert row_a["samples_per_sec"] == pytest.approx(12.0)

    def test_generalization_gap(self, tables: dict[str, pd.DataFrame]) -> None:
        result = summarize_runs(tables)
        row_a = result[result["run_uuid"] == RUN_A].iloc[0]
        expected_gap = pytest.approx(0.85 - 0.80)
        assert row_a["generalization_gap_final"] == expected_gap

    def test_best_val_dice_epoch_correct(self, tables: dict[str, pd.DataFrame]) -> None:
        # RUN_A val_dice: [0.70, 0.80, 0.75] → best at epoch 2
        result = summarize_runs(tables)
        row_a = result[result["run_uuid"] == RUN_A].iloc[0]
        assert row_a["best_val_dice_epoch"] == 2

    def test_min_val_loss_epoch_correct(self, tables: dict[str, pd.DataFrame]) -> None:
        # RUN_A val_loss: [0.50, 0.35, 0.40] → min at epoch 2
        result = summarize_runs(tables)
        row_a = result[result["run_uuid"] == RUN_A].iloc[0]
        assert row_a["min_val_loss_epoch"] == 2

    def test_missing_latest_key_gives_nan(self, tables: dict[str, pd.DataFrame]) -> None:
        # Remove all final/val_iou entries
        tables["latest"] = tables["latest"][tables["latest"]["key"] != "final/val_iou"]
        result = summarize_runs(tables)
        assert result["final_val_iou"].isna().all()

    def test_no_artifact_root_no_checkpoint_columns(self, tables: dict[str, pd.DataFrame]) -> None:
        result = summarize_runs(tables, artifact_root=None)
        assert "checkpoint" not in result.columns
        assert "model_total_params" not in result.columns

    def test_artifact_root_missing_checkpoint_dir(self, tables: dict[str, pd.DataFrame], tmp_path: Path) -> None:
        # Directory exists but contains no .ckpt files → checkpoint=None, checkpoint_mb=NaN
        result = summarize_runs(tables, artifact_root=tmp_path)
        assert result["checkpoint"].isna().all()
        assert result["checkpoint_mb"].isna().all()

    def test_artifact_root_finds_checkpoint(self, tables: dict[str, pd.DataFrame], tmp_path: Path) -> None:
        ckpt_dir = tmp_path / "1" / RUN_A / "artifacts" / "checkpoints" / "best"
        ckpt_dir.mkdir(parents=True)
        ckpt_file = ckpt_dir / "epoch=2-step=100.ckpt"
        ckpt_file.write_bytes(b"x" * 5_000_000)
        result = summarize_runs(tables, artifact_root=tmp_path)
        row_a = result[result["run_uuid"] == RUN_A].iloc[0]
        assert row_a["checkpoint"] == "epoch=2-step=100.ckpt"
        assert row_a["checkpoint_mb"] == pytest.approx(5.0)

    def test_artifact_root_reads_model_params(self, tables: dict[str, pd.DataFrame], tmp_path: Path) -> None:
        model_dir = tmp_path / "1" / RUN_A / "artifacts" / "model"
        model_dir.mkdir(parents=True)
        # _model_total_params returns line.split("Total params")[0].strip()
        summary = textwrap.dedent("""\
            Layer (type)        Output Shape
            ----------------------------------
            3,700,000 Total params
        """)
        (model_dir / "model_summary.txt").write_text(summary)
        result = summarize_runs(tables, artifact_root=tmp_path)
        row_a = result[result["run_uuid"] == RUN_A].iloc[0]
        assert row_a["model_total_params"] == "3,700,000"

    def test_empty_runs_returns_empty_dataframe(self) -> None:
        tables = {
            "runs": pd.DataFrame(columns=["run_uuid", "experiment_id", "experiment_name", "status", "start_time", "end_time"]),
            "metrics": pd.DataFrame(columns=["run_uuid", "key", "value", "step", "timestamp"]),
            "latest": pd.DataFrame(columns=["run_uuid", "key", "value"]),
        }
        result = summarize_runs(tables)
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
# family_summary
# ---------------------------------------------------------------------------

class TestFamilySummary:
    def _run_summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "run_uuid": [RUN_A, RUN_B],
                "encoder": ["resnet50", "efficientnet"],
                "merge": ["add", "concat"],
                "best_val_best_dice_at_threshold": [0.81, 0.77],
                "final_val_best_dice_at_threshold": [0.80, 0.76],
                "samples_per_sec": [12.0, 11.0],
                "generalization_gap_final": [0.05, 0.02],
            }
        )

    def test_contains_encoder_and_merge_families(self) -> None:
        result = family_summary(self._run_summary()).data
        assert set(result["family"]) == {"encoder", "merge"}

    def test_sorted_within_family_descending_dice(self) -> None:
        summary = self._run_summary()
        summary = pd.concat([summary, summary.assign(
            run_uuid=["rx", "ry"], encoder=["resnet50", "resnet50"],
            best_val_best_dice_at_threshold=[0.90, 0.70]
        )], ignore_index=True)
        result = family_summary(summary, group_cols=["encoder"]).data
        encoder_rows = result[result["family"] == "encoder"].reset_index(drop=True)
        assert encoder_rows.iloc[0]["mean_best_dice"] >= encoder_rows.iloc[1]["mean_best_dice"]

    def test_n_counts_runs_per_group(self) -> None:
        result = family_summary(self._run_summary(), group_cols=["encoder"]).data
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
# rank_table
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
            "best_val_best_dice_at_threshold": [0.81, 0.85],
            "best_val_best_dice_at_threshold_tail_mean": [0.80, 0.77],
            "best_val_best_dice_at_threshold_epoch": [2, 3],
            "best_val_dice": [0.80, 0.84],
            "best_val_iou": [0.72, 0.75],
            "min_val_loss": [0.30, 0.28],
            "final_val_best_dice_at_threshold": [0.80, 0.83],
            "final_val_dice": [0.79, 0.82],
            "generalization_gap_final": [0.05, 0.04],
            "samples_per_sec": [12.0, 11.0],
            "duration_min": [2.0, 1.0],
        })

    def test_returns_styler(self) -> None:
        from pandas.io.formats.style import Styler
        assert isinstance(rank_table(self._run_summary()), Styler)

    def test_sort_by_best_puts_highest_peak_first(self) -> None:
        # RUN_B peak=0.85 > RUN_A peak=0.81
        df = rank_table(self._run_summary(), sort_by="best").data
        assert df.iloc[0]["best_val_best_dice_at_threshold"] == pytest.approx(0.85)

    def test_sort_by_tail_puts_highest_tail_mean_first(self) -> None:
        # RUN_A tail_mean=0.80 > RUN_B tail_mean=0.77
        df = rank_table(self._run_summary(), sort_by="tail").data
        assert df.iloc[0]["best_val_best_dice_at_threshold_tail_mean"] == pytest.approx(0.80)

    def test_invalid_sort_by_raises(self) -> None:
        with pytest.raises(ValueError, match="sort_by"):
            rank_table(self._run_summary(), sort_by="median")

    def test_caption_best_names_sort_col_and_criterion(self) -> None:
        caption = rank_table(self._run_summary(), sort_by="best").caption
        assert "best_val_best_dice_at_threshold" in caption
        assert "peak value" in caption

    def test_caption_tail_names_sort_col_and_window(self) -> None:
        caption = rank_table(self._run_summary(), sort_by="tail", tail_n=5).caption
        assert "best_val_best_dice_at_threshold_tail_mean" in caption
        assert "5" in caption
