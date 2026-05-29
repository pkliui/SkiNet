"""Reusable MLflow analysis helpers  - IO
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import pandas as pd


def load_mlflow_tables(db_path: Path) -> dict[str, pd.DataFrame]:
    """
    Load data from an MLflow tracking database (SQLite) into a dictionary of pandas DataFrames.

    Use default structure of an MLflow SQLite store, with tables for experiments, runs, params,
    metrics, and latest_metrics.

    Join experiments and runs on experiment_id.
    """
    queries = {
        "experiments": """
            select experiment_id, name, lifecycle_stage, creation_time, last_update_time
            from experiments
            order by experiment_id
        """,
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
    with sqlite3.connect(db_path) as con:
        return {name: pd.read_sql_query(sql, con) for name, sql in queries.items()}
