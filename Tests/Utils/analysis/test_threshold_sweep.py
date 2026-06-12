"""Unit tests for SkiNet.Utils.analysis.threshold_sweep."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from SkiNet.Utils.analysis.threshold_sweep import (
    DICE_05_KEY,
    GAIN_KEY,
    IOU_KEY,
    SPS_KEY,
    SWEPT_KEY,
    TAU_KEY,
    TRAIN_DICE_KEY,
    epoch_trajectories,
    load_threshold_sweep,
    paired_gain_stats,
    threshold_stability,
)

# ---------------------------------------------------------------------------
# Synthetic-run construction
# ---------------------------------------------------------------------------

# Two seeds, four epochs each. swept (val_best_dice_at_threshold) peaks at a
# known epoch per seed so the best-epoch selection is hand-checkable:
#   seed 100 — swept peaks at epoch 3 (step 2)
#   seed 101 — swept peaks at epoch 2 (step 1)
_RUNS = {
    100: {
        "run_uuid": "seed100-uuid",
        SWEPT_KEY: [0.80, 0.83, 0.88, 0.85],   # argmax -> epoch 3 (0-based step 2)
        DICE_05_KEY: [0.78, 0.80, 0.84, 0.82],   # gain at best = 0.88 - 0.84 = 0.04
        TAU_KEY: [0.20, 0.45, 0.55, 0.30],   # tau* at best = 0.55
        GAIN_KEY: [0.02, 0.03, 0.04, 0.03],
        IOU_KEY: [0.70, 0.72, 0.76, 0.74],   # iou at best = 0.76
        TRAIN_DICE_KEY: [0.86, 0.88, 0.90, 0.91],   # gen_gap at best = 0.90 - 0.84 = 0.06
        SPS_KEY: [100.0, 102.0, 104.0, 106.0],  # mean = 103.0
    },
    101: {
        "run_uuid": "seed101-uuid",
        SWEPT_KEY: [0.81, 0.90, 0.86, 0.84],   # argmax -> epoch 2 (step 1)
        DICE_05_KEY: [0.79, 0.86, 0.83, 0.82],   # gain at best = 0.90 - 0.86 = 0.04
        TAU_KEY: [0.10, 0.40, 0.60, 0.50],   # tau* at best = 0.40
        GAIN_KEY: [0.02, 0.04, 0.03, 0.02],
        IOU_KEY: [0.71, 0.78, 0.75, 0.74],
        TRAIN_DICE_KEY: [0.87, 0.93, 0.92, 0.93],   # gen_gap at best = 0.93 - 0.86 = 0.07
        SPS_KEY: [110.0, 112.0, 114.0, 116.0],  # mean = 113.0
    },
}


_SCHEMA_SQL = """
CREATE TABLE experiments (
    experiment_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL DEFAULT 'Default',
    artifact_location TEXT,
    lifecycle_stage TEXT NOT NULL DEFAULT 'active'
);
INSERT INTO experiments VALUES (0, 'Default', '', 'active');
CREATE TABLE runs (
    run_uuid TEXT PRIMARY KEY, name TEXT, status TEXT,
    start_time INTEGER, end_time INTEGER, artifact_uri TEXT,
    experiment_id INTEGER DEFAULT 0,
    lifecycle_stage TEXT NOT NULL DEFAULT 'active'
);
CREATE TABLE params (run_uuid TEXT, key TEXT, value TEXT);
CREATE TABLE metrics (
    run_uuid TEXT, key TEXT, value REAL, step INTEGER, timestamp INTEGER
);
CREATE TABLE latest_metrics (
    run_uuid TEXT, key TEXT, value REAL, step INTEGER, timestamp INTEGER
);
"""


def _create_sweep_db(path: Path) -> None:
    """Write the two synthetic sweep runs above into a minimal MLflow store."""
    with closing(sqlite3.connect(path)) as con:
        con.executescript(_SCHEMA_SQL)
        for seed, run in _RUNS.items():
            con.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                (run["run_uuid"], f"run_seed{seed}", "FINISHED", 0, 0, "", 0, "active"),
            )
            con.execute(
                "INSERT INTO params VALUES (?,?,?)",
                (run["run_uuid"], "seed", str(seed)),
            )
            for key, series in run.items():
                if key == "run_uuid":
                    continue
                con.executemany(
                    "INSERT INTO metrics VALUES (?,?,?,?,?)",
                    [(run["run_uuid"], key, v, step, step) for step, v in enumerate(series)],
                )
                last_step = len(series) - 1
                con.execute(
                    "INSERT INTO latest_metrics VALUES (?,?,?,?,?)",
                    (run["run_uuid"], key, series[-1], last_step, last_step),
                )
        con.commit()


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "sweep.db"
    _create_sweep_db(p)
    return p


@pytest.fixture()
def per_seed(db_path: Path) -> pd.DataFrame:
    # warmup=0 so the 4-epoch synthetic runs are not emptied by the wander window.
    return load_threshold_sweep(db_path, part_split=100, warmup=0)


# ---------------------------------------------------------------------------
# load_threshold_sweep
# ---------------------------------------------------------------------------

class TestLoadThresholdSweep:
    def test_one_row_per_seed_sorted(self, per_seed: pd.DataFrame) -> None:
        assert per_seed["seed"].tolist() == [100, 101]

    def test_best_epoch_is_swept_argmax(self, per_seed: pd.DataFrame) -> None:
        # 1-based: seed 100 peaks at epoch 3, seed 101 at epoch 2.
        assert per_seed.set_index("seed")["best_ep"].to_dict() == {100: 3, 101: 2}

    def test_swept_and_dice_read_at_best_epoch(self, per_seed: pd.DataFrame) -> None:
        s100 = per_seed[per_seed["seed"] == 100].iloc[0]
        assert s100["val_best_dice_at_threshold"] == pytest.approx(0.88)
        assert s100["val_dice"] == pytest.approx(0.84)

    def test_gain_is_swept_minus_dice(self, per_seed: pd.DataFrame) -> None:
        assert per_seed["val_dice_gain"].tolist() == pytest.approx([0.04, 0.04])

    def test_tau_star_read_at_best_epoch(self, per_seed: pd.DataFrame) -> None:
        assert per_seed.set_index("seed")["val_optimal_threshold"].to_dict() == pytest.approx(
            {100: 0.55, 101: 0.40}
        )

    def test_companion_metrics_at_best_epoch(self, per_seed: pd.DataFrame) -> None:
        s100 = per_seed[per_seed["seed"] == 100].iloc[0]
        assert s100["val_iou"] == pytest.approx(0.76)
        assert s100["gen_gap"] == pytest.approx(0.06)        # 0.90 - 0.84
        assert s100["samples_per_sec"] == pytest.approx(103.0)  # mean over 4 epochs

    def test_part_split_labels(self, per_seed: pd.DataFrame) -> None:
        assert per_seed.set_index("seed")["part"].to_dict() == {100: "P1", 101: "P2"}

    def test_part_split_none_all_p1(self, db_path: Path) -> None:
        df = load_threshold_sweep(db_path, warmup=0)
        assert set(df["part"]) == {"P1"}

    def test_wander_window_respects_warmup(self, db_path: Path) -> None:
        # warmup=3 leaves only the final epoch (step 3) -> sd over a single point is 0.
        df = load_threshold_sweep(db_path, warmup=3)
        assert df["val_tau_wander_sd"].tolist() == pytest.approx([0.0, 0.0])
        # val_tau_min == val_tau_max == the last-epoch tau for each seed.
        assert df.set_index("seed")["val_tau_min"].to_dict() == pytest.approx({100: 0.30, 101: 0.50})

    def test_wander_sd_over_full_history(self, per_seed: pd.DataFrame) -> None:
        # warmup=0 -> wander sd is population std (ddof=0) over all 4 tau values.
        expected = float(np.std(np.array(_RUNS[100][TAU_KEY], dtype=float)))
        s100 = per_seed[per_seed["seed"] == 100].iloc[0]
        assert s100["val_tau_wander_sd"] == pytest.approx(expected)

    def test_concatenates_disjoint_dbs(self, db_path: Path, tmp_path: Path) -> None:
        # A second store with a distinct seed (102) must append, not collide.
        p2 = tmp_path / "sweep_part2.db"
        with closing(sqlite3.connect(p2)) as con:
            con.executescript(_SCHEMA_SQL)
            con.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                ("seed102-uuid", "run_seed102", "FINISHED", 0, 0, "", 0, "active"),
            )
            con.execute("INSERT INTO params VALUES (?,?,?)", ("seed102-uuid", "seed", "102"))
            for key, series in _RUNS[100].items():
                if key == "run_uuid":
                    continue
                con.executemany(
                    "INSERT INTO metrics VALUES (?,?,?,?,?)",
                    [("seed102-uuid", key, v, step, step) for step, v in enumerate(series)],
                )
                last_step = len(series) - 1
                con.execute(
                    "INSERT INTO latest_metrics VALUES (?,?,?,?,?)",
                    ("seed102-uuid", key, series[-1], last_step, last_step),
                )
            con.commit()
        df = load_threshold_sweep(db_path, p2, part_split=100, warmup=0)
        assert df["seed"].tolist() == [100, 101, 102]
        assert df.set_index("seed")["part"].to_dict() == {100: "P1", 101: "P2", 102: "P2"}

    def test_empty_db_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.db"
        with closing(sqlite3.connect(p)) as con:
            con.executescript(_SCHEMA_SQL)
        with pytest.raises(ValueError, match="no active runs or no metrics"):
            load_threshold_sweep(p)


# ---------------------------------------------------------------------------
# epoch_trajectories
# ---------------------------------------------------------------------------

class TestEpochTrajectories:
    def test_returns_full_series_per_seed(self, db_path: Path) -> None:
        traj = epoch_trajectories(db_path, keys=(TAU_KEY, GAIN_KEY), part_split=100)
        assert set(traj) == {100, 101}
        np.testing.assert_allclose(traj[100][TAU_KEY], np.array(_RUNS[100][TAU_KEY], dtype=float))
        np.testing.assert_allclose(traj[101][GAIN_KEY], np.array(_RUNS[101][GAIN_KEY], dtype=float))

    def test_part_label_attached(self, db_path: Path) -> None:
        traj = epoch_trajectories(db_path, keys=(TAU_KEY,), part_split=100)
        assert traj[100]["part"] == "P1"
        assert traj[101]["part"] == "P2"


# ---------------------------------------------------------------------------
# paired_gain_stats
# ---------------------------------------------------------------------------

class TestPairedGainStats:
    def test_mean_and_count(self) -> None:
        # Use a 4-seed frame with non-zero variance so BCa is well-defined.
        df = pd.DataFrame({
            "val_dice_gain": [0.01, 0.03, 0.02, 0.04],
            "val_best_dice_at_threshold": [0.81, 0.83, 0.82, 0.84],
            "val_dice": [0.80, 0.80, 0.80, 0.80],
        })
        st = paired_gain_stats(df)
        assert st["n"] == 4
        assert st["mean"] == pytest.approx(0.025)
        assert st["n_positive"] == 4

    def test_zero_variance_raises(self, per_seed: pd.DataFrame) -> None:
        # Both gains identical (sd = 0) → BCa is degenerate; the function must
        # raise rather than silently return NaN bounds.
        with pytest.raises(ValueError, match="BCa CI is NaN"):
            paired_gain_stats(per_seed)

    def test_known_dz_on_varied_gains(self) -> None:
        df = pd.DataFrame(
            {
                "val_dice_gain": [0.01, 0.03, 0.02, 0.04],
                "val_best_dice_at_threshold": [0.81, 0.83, 0.82, 0.84],
                "val_dice": [0.80, 0.80, 0.80, 0.80],
            }
        )
        st = paired_gain_stats(df)
        g = np.array([0.01, 0.03, 0.02, 0.04])
        assert st["mean"] == pytest.approx(g.mean())
        assert st["cohen_dz"] == pytest.approx(g.mean() / g.std(ddof=1))


# ---------------------------------------------------------------------------
# threshold_stability
# ---------------------------------------------------------------------------

class TestThresholdStability:
    def test_median_and_distance_from_half(self, per_seed: pd.DataFrame) -> None:
        stab = threshold_stability(per_seed)
        # tau* = [0.55, 0.40] -> median 0.475, distance from 0.5 = 0.025
        assert stab["tau_median"] == pytest.approx(0.475)
        assert stab["tau_dist_from_half"] == pytest.approx(0.025)

    def test_global_span_and_mean_wander(self, per_seed: pd.DataFrame) -> None:
        stab = threshold_stability(per_seed)
        assert stab["tau_global_min"] == pytest.approx(0.10)  # min over all tail tau
        assert stab["tau_global_max"] == pytest.approx(0.60)
        assert stab["mean_wander_sd"] == pytest.approx(per_seed["val_tau_wander_sd"].mean())
