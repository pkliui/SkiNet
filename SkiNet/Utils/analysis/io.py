"""Reusable MLflow analysis helpers  - IO
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
import pandas as pd


def load_mlflow_tables(db_path: Path) -> dict[str, pd.DataFrame]:
    """
    Load four analysis-relevant tables from an MLflow SQLite tracking store.
    Return a dict keyed by table name, where each value is a DataFrame containing the selected
    columns from that table.

    Tables returned:

    - "runs": one row per training run, joined with experiments so each row carries
      both run_name (e.g. "run_seed105", unique per run) and experiment_name (shared
      across all runs in the same group). Filtered to lifecycle_stage = 'active' -
      deleted runs are never visible to callers. This is the backbone table —
      run_uuid is the join key used by every other table.

    - "params": logged hyperparameters in long format — one row per (run, param name),
      e.g. (uuid, "lr", "3e-4"). This means each run occupies multiple rows, one per
      hyperparameter. If you need one row per run with each param as its own column,
      pivot with df.pivot(index="run_uuid", columns="key", values="value").

    - "metrics": full training curves — one row per (run, metric, step). Can be large
      (n_runs x n_metrics x n_epochs). Use when you need per-epoch trajectories.

    - "latest": final value only of each metric per run — one row per (run, metric).
      MLflow maintains this as a materialised snapshot, so it is much smaller than
      "metrics". Use this as the fast path when only end-of-training numbers are needed.
    """
    queries = {
        "runs": """
            select
                r.run_uuid,
                r.name as run_name,
                r.status,
                r.start_time,
                r.end_time,
                r.artifact_uri,
                r.experiment_id,
                e.name as experiment_name
            from runs r
            left join experiments e on e.experiment_id = r.experiment_id
            where r.lifecycle_stage = 'active'
            order by r.experiment_id
        """,
        "params": "select run_uuid, key, value from params",
        "metrics": "select run_uuid, key, value, step, timestamp from metrics",
        "latest": "select run_uuid, key, value, step, timestamp from latest_metrics",
    }
    with closing(sqlite3.connect(db_path)) as con:
        tables = {name: pd.read_sql_query(sql, con) for name, sql in queries.items()}
    empty = [name for name, df in tables.items() if df.empty]
    if empty:
        raise ValueError(
            f"no active runs or no metrics in {db_path!r} (empty tables: {empty}). "
            "Ensure the tracking store has active runs with logged params and metrics."
        )
    return tables
