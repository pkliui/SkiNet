import sqlite3
import time
from pathlib import Path

import pandas as pd
import pytest

from SkiNet.Utils.analysis.io import load_mlflow_tables


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> int:
    """Return a stable integer timestamp (ms)."""
    return int(time.time() * 1000)


def create_mlflow_db(path: Path, *, include_deleted_run: bool = False) -> None:
    """Create a minimal MLflow-compatible SQLite database at *path*."""
    with sqlite3.connect(path) as con:
        cur = con.cursor()

        cur.executescript("""
            CREATE TABLE experiments (
                experiment_id   INTEGER PRIMARY KEY,
                name            TEXT NOT NULL,
                lifecycle_stage TEXT NOT NULL DEFAULT 'active',
                creation_time   INTEGER,
                last_update_time INTEGER
            );

            CREATE TABLE runs (
                run_uuid        TEXT PRIMARY KEY,
                name            TEXT,
                status          TEXT,
                start_time      INTEGER,
                end_time        INTEGER,
                artifact_uri    TEXT,
                experiment_id   INTEGER,
                lifecycle_stage TEXT NOT NULL DEFAULT 'active'
            );

            CREATE TABLE params (
                run_uuid TEXT,
                key      TEXT,
                value    TEXT
            );

            CREATE TABLE metrics (
                run_uuid  TEXT,
                key       TEXT,
                value     REAL,
                step      INTEGER,
                timestamp INTEGER
            );

            CREATE TABLE latest_metrics (
                run_uuid  TEXT,
                key       TEXT,
                value     REAL,
                step      INTEGER,
                timestamp INTEGER
            );
        """)
        # Capture the current millisecond timestamp once
        now = _ts()

        # Two experiments
        cur.executemany(
            "INSERT INTO experiments VALUES (?,?,?,?,?)",
            [
                (1, "exp-alpha", "active", now, now),
                (2, "exp-beta", "active", now, now),
            ],
        )

        # Active runs
        cur.executemany(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
            [
                ("run-aaa", "run-1", "FINISHED", now, now, "s3://a/1", 1, "active"),
                ("run-bbb", "run-2", "RUNNING", now, None, "s3://a/2", 2, "active"),
            ],
        )

        if include_deleted_run:
            cur.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                ("run-del", "deleted-run", "FINISHED", now, now, "s3://a/d", 1, "deleted"),
            )

        # Params
        cur.executemany(
            "INSERT INTO params VALUES (?,?,?)",
            [
                ("run-aaa", "lr", "0.01"),
                ("run-aaa", "epochs", "10"),
                ("run-bbb", "lr", "0.001"),
            ],
        )

        # Metrics  (two steps for run-aaa)
        cur.executemany(
            "INSERT INTO metrics VALUES (?,?,?,?,?)",
            [
                ("run-aaa", "loss", 0.9, 0, now),
                ("run-aaa", "loss", 0.5, 1, now),
                ("run-bbb", "loss", 0.8, 0, now),
            ],
        )

        # Latest metrics
        cur.executemany(
            "INSERT INTO latest_metrics VALUES (?,?,?,?,?)",
            [
                ("run-aaa", "loss", 0.5, 1, now),
                ("run-bbb", "loss", 0.8, 0, now),
            ],
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Fixture to create an MLflow database and return its path."""
    p = tmp_path / "mlflow.db"
    create_mlflow_db(p)
    return p


@pytest.fixture()
def db_path_with_deleted(tmp_path: Path) -> Path:
    """Fixture to create an MLflow database that includes a deleted run."""
    p = tmp_path / "mlflow_deleted.db"
    create_mlflow_db(p, include_deleted_run=True)
    return p


@pytest.fixture()
def tables(db_path: Path) -> dict[str, pd.DataFrame]:
    """Fixture to load MLflow tables from *db_path*."""
    return load_mlflow_tables(db_path)


# ---------------------------------------------------------------------------
# Return-value structure
# ---------------------------------------------------------------------------

class TestReturnStructure:
    """Test that load_mlflow_tables() returns a dict with the expected keys (as per MLflow)
    and all values are DataFrames."""
    EXPECTED_KEYS = {"experiments", "runs", "params", "metrics", "latest"}

    def test_returns_dict(self, tables: dict[str, pd.DataFrame]) -> None:
        assert isinstance(tables, dict)

    def test_has_all_keys(self, tables: dict[str, pd.DataFrame]) -> None:
        assert tables.keys() == self.EXPECTED_KEYS

    def test_all_values_are_dataframes(self, tables: dict[str, pd.DataFrame]) -> None:
        for key, df in tables.items():
            assert isinstance(df, pd.DataFrame), f"{key!r} is not a DataFrame"


# ---------------------------------------------------------------------------
# experiments table
# ---------------------------------------------------------------------------

class TestExperiments:
    def test_row_count(self, tables: dict[str, pd.DataFrame]) -> None:
        assert len(tables["experiments"]) == 2

    def test_columns(self, tables: dict[str, pd.DataFrame]) -> None:
        expected = {"experiment_id", "name", "lifecycle_stage", "creation_time", "last_update_time"}
        assert set(tables["experiments"].columns) == expected

    def test_ordered_by_experiment_id(self, tables: dict[str, pd.DataFrame]) -> None:
        """Verify the ORDER BY experiment_id clause in the query is actually respected."""
        ids = tables["experiments"]["experiment_id"].tolist()
        assert ids == sorted(ids)

    def test_names(self, tables: dict[str, pd.DataFrame]) -> None:
        assert set(tables["experiments"]["name"]) == {"exp-alpha", "exp-beta"}


# ---------------------------------------------------------------------------
# runs table
# ---------------------------------------------------------------------------

class TestRuns:
    def test_row_count(self, tables: dict[str, pd.DataFrame]) -> None:
        assert len(tables["runs"]) == 2

    def test_columns(self, tables: dict[str, pd.DataFrame]) -> None:
        expected = {
            "run_uuid", "run_name", "status", "start_time", "end_time",
            "artifact_uri", "experiment_id", "experiment_name",
        }
        assert set(tables["runs"].columns) == expected

    def test_experiment_name_joined(self, tables: dict[str, pd.DataFrame]) -> None:
        mapping = dict(zip(tables["runs"]["run_uuid"], tables["runs"]["experiment_name"]))
        assert mapping["run-aaa"] == "exp-alpha"
        assert mapping["run-bbb"] == "exp-beta"

    def test_deleted_runs_excluded(self, db_path_with_deleted: Path) -> None:
        tables = load_mlflow_tables(db_path_with_deleted)
        assert "run-del" not in tables["runs"]["run_uuid"].values

    def test_ordered_by_experiment_id(self, tables: dict[str, pd.DataFrame]) -> None:
        ids = tables["runs"]["experiment_id"].tolist()
        assert ids == sorted(ids)

    def test_statuses_present(self, tables: dict[str, pd.DataFrame]) -> None:
        assert set(tables["runs"]["status"]) == {"FINISHED", "RUNNING"}


# ---------------------------------------------------------------------------
# params table
# ---------------------------------------------------------------------------

class TestParams:
    def test_row_count(self, tables: dict[str, pd.DataFrame]) -> None:
        assert len(tables["params"]) == 3

    def test_columns(self, tables: dict[str, pd.DataFrame]) -> None:
        assert set(tables["params"].columns) == {"run_uuid", "key", "value"}

    def test_values(self, tables: dict[str, pd.DataFrame]) -> None:
        aaa_params = (
            tables["params"]
            .loc[tables["params"]["run_uuid"] == "run-aaa"]
            .set_index("key")["value"]
            .to_dict()
        )
        assert aaa_params == {"lr": "0.01", "epochs": "10"}


# ---------------------------------------------------------------------------
# metrics table
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_row_count(self, tables: dict[str, pd.DataFrame]) -> None:
        assert len(tables["metrics"]) == 3

    def test_columns(self, tables: dict[str, pd.DataFrame]) -> None:
        assert set(tables["metrics"].columns) == {"run_uuid", "key", "value", "step", "timestamp"}

    def test_multiple_steps_for_same_key(self, tables: dict[str, pd.DataFrame]) -> None:
        aaa_loss = tables["metrics"][
            (tables["metrics"]["run_uuid"] == "run-aaa") & (tables["metrics"]["key"] == "loss")
        ]
        assert len(aaa_loss) == 2
        assert set(aaa_loss["step"]) == {0, 1}


# ---------------------------------------------------------------------------
# latest_metrics table
# ---------------------------------------------------------------------------

class TestLatestMetrics:
    """
    Verify that latest_metrics contains the latest value for each (run_uuid, key) pair, as per
    https://mlflow.org/docs/latest/tracking.html#latest-metrics
    """

    def test_row_count(self, tables: dict[str, pd.DataFrame]) -> None:
        assert len(tables["latest"]) == 2

    def test_columns(self, tables: dict[str, pd.DataFrame]) -> None:
        assert set(tables["latest"].columns) == {"run_uuid", "key", "value", "step", "timestamp"}

    def test_latest_value_for_run_aaa(self, tables: dict[str, pd.DataFrame]) -> None:
        row = tables["latest"][tables["latest"]["run_uuid"] == "run-aaa"].iloc[0]
        assert row["value"] == pytest.approx(0.5)
        assert row["step"] == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_tables(self, tmp_path: Path) -> None:
        """Function should return empty DataFrames, not raise, when tables are empty."""
        p = tmp_path / "empty.db"
        # Build schema only — no rows
        with sqlite3.connect(p) as con:
            con.executescript("""
                CREATE TABLE experiments (experiment_id INTEGER PRIMARY KEY, name TEXT,
                    lifecycle_stage TEXT, creation_time INTEGER, last_update_time INTEGER);
                CREATE TABLE runs (run_uuid TEXT PRIMARY KEY, name TEXT, status TEXT,
                    start_time INTEGER, end_time INTEGER, artifact_uri TEXT,
                    experiment_id INTEGER, lifecycle_stage TEXT DEFAULT 'active');
                CREATE TABLE params (run_uuid TEXT, key TEXT, value TEXT);
                CREATE TABLE metrics (run_uuid TEXT, key TEXT, value REAL,
                    step INTEGER, timestamp INTEGER);
                CREATE TABLE latest_metrics (run_uuid TEXT, key TEXT, value REAL,
                    step INTEGER, timestamp INTEGER);
            """)
        tables = load_mlflow_tables(p)
        for key in ("experiments", "runs", "params", "metrics", "latest"):
            assert len(tables[key]) == 0, f"Expected {key!r} to be empty"

    def test_accepts_path_object(self, db_path: Path) -> None:
        """db_path should work as a pathlib.Path (not just a string)."""
        result = load_mlflow_tables(db_path)
        assert "experiments" in result

    def test_invalid_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):
            load_mlflow_tables(tmp_path / "nonexistent.db")
