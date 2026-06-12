"""Batch size sweep analysis — data loading, outlier marking, and summary helpers.

These functions support E0-style throughput sweep notebooks: they load step-level
metric series from MLflow SQLite DBs, mark the known artefact outliers, and compute
per-batch-size throughput/GPU summaries and recommendation facts.

All public functions operate on a tidy long-format ``DataFrame`` with the column
schema defined in :data:`~SkiNet.Utils.analysis.schema.BATCH_SWEEP_COLS`.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from SkiNet.Utils.analysis.schema import (
    EXPECTED_BATCH_SIZES,
    MAX_EPOCHS_SWEEP,
    N_TRAIN_IMAGES_ISIC2017,
    BATCH_SWEEP_METRICS,
    BATCH_SWEEP_COLS,
)


# ── SQLite loaders ─────────────────────────────────────────────────────────────

def fetch_run_batch_map(con: sqlite3.Connection,
                        batch_sizes: list[int] = EXPECTED_BATCH_SIZES) -> pd.DataFrame:
    """Return a ``[run_uuid, batch_size]`` lookup for all active runs with a logged batch size.

    :param con: Open SQLite connection to an MLflow tracking store.
    :param batch_sizes: Whitelist of batch sizes to retain; others are discarded.
    :return: Two-column DataFrame ``[run_uuid, batch_size]`` with ``batch_size`` as int.
    """
    q = """
        SELECT r.run_uuid, p.value AS batch_size
        FROM runs r
        JOIN params p ON p.run_uuid = r.run_uuid AND p.key = 'batch_size'
        WHERE r.lifecycle_stage = 'active'
    """
    run_map = pd.read_sql(q, con)
    run_map["batch_size"] = run_map["batch_size"].astype(int)
    return run_map[run_map["batch_size"].isin(batch_sizes)].reset_index(drop=True)


def load_metric_series(con: sqlite3.Connection,
                       run_uuid: str,
                       key: str) -> pd.DataFrame:
    """Fetch every logged value for one metric key in one run, sorted by step then timestamp.

    :param con: Open SQLite connection.
    :param run_uuid: MLflow run identifier.
    :param key: Metric key as logged by Lightning/MLflow, e.g. ``"perf/samples_per_sec"``,
        ``"perf/time_per_step_ms"``, ``"train_loss"``, ``"val_best_dice_at_threshold"``,
        ``"val_mean_dice_per_image"``, ``"val_dice_threshold_gain"``, or ``"grad_scale"``.
    :return: DataFrame ``[step, value, timestamp]``. Empty if the key was never logged.
    """
    q = """
        SELECT step, value, timestamp
        FROM metrics
        WHERE run_uuid = ? AND key = ?
        ORDER BY step ASC, timestamp ASC
    """
    return pd.read_sql(q, con, params=(run_uuid, key))


def build_df_batch_sweep_run(con: sqlite3.Connection,
                             run_uuid: str,
                             batch_size: int,
                             experiment: str,
                             batch_sweep_metrics: list[str] = BATCH_SWEEP_METRICS,
                             batch_sweep_cols: list[str] = BATCH_SWEEP_COLS) -> pd.DataFrame:
    """Assemble one batch sweep run's metrics into a single tidy DataFrame.

    Each metric is deduplicated to the first logged entry per step before merging.
    Missing metric columns (e.g. GPU metrics not logged in older runs) are filled
    with ``NaN`` so the output schema is always ``BATCH_SWEEP_COLS``.

    :param con: Open SQLite connection to the MLflow tracking store.
    :param run_uuid: MLflow run identifier.
    :param batch_size: Batch size for this run, read from ``params`` by the caller.
    :param experiment: Experiment label written into the ``experiment`` column, e.g. ``"no_aug"``.
    :param batch_sweep_metrics: MLflow metric keys for the batch sweep run to fetch; unmapped keys are silently skipped.
    :param batch_sweep_cols: Expected output column order; empty frame returned when no metrics found.
    :return: Tidy DataFrame conforming to ``batch_sweep_cols`` with ``is_outlier=False``
             and ``outlier_reason=""`` — outlier flags are set later by :func:`mark_outliers`.
    """
    metric_dfs: dict[str, pd.DataFrame] = {}
    for key in batch_sweep_metrics:
        df = load_metric_series(con, run_uuid, key)
        if df.empty:
            continue
        metric_dfs[key] = (df.sort_values(["step", "timestamp"])
                           .drop_duplicates(subset=["step"], keep="first")
                           .rename(columns={"value": key})
                           [["step", key]])

    if not metric_dfs:
        return pd.DataFrame(columns=batch_sweep_cols)

    # Merge all metric DataFrames on "step" using an outer join, so missing metrics produce NaN.
    out = list(metric_dfs.values())[0]
    for df in list(metric_dfs.values())[1:]:
        out = out.merge(df, on="step", how="outer")

    out = out.rename(columns={
        "perf/samples_per_sec": "samples_per_sec",
        "perf/time_per_step_ms": "time_per_step_ms",
        "system/gpu_mem_allocated_gb": "gpu_mem_gb",
        "system/gpu_util_percent": "gpu_util_pct",
        "train_loss_step": "train_loss",
        "epoch": "epoch_idx",
    }).assign(experiment=experiment, batch_size=batch_size, run_uuid=run_uuid,
              is_outlier=False, outlier_reason="")
    return out.reindex(columns=batch_sweep_cols)


# ── Outlier marking ────────────────────────────────────────────────────────────

def get_rule_outlier_steps(steps_per_epoch: int,
                           max_epochs: int = MAX_EPOCHS_SWEEP) -> dict[int, str]:
    """Return a ``{step_index: reason}`` dict for every step that is a known structural artefact.

    Three artefact types are flagged regardless of their measured value:

    - ``"step0"`` — step 0 always has a duplicate entry (CUDA warm-up at training start
      and an end-of-run summary that MLflow writes back to step 0)
    - ``"epoch_last_high"`` — the final step of each epoch (``n * steps_per_epoch - 1``)
      is slow because a checkpoint is saved there
    - ``"epoch_first_low"`` — the first step of the next epoch (``n * steps_per_epoch``)
      is slow because the data loader cache is cold after the checkpoint pause

    ``setdefault`` is used so that if two boundaries coincide on the same step index
    the first label written wins rather than being overwritten.

    :param steps_per_epoch: Unique step count divided by epoch count (``nunique // max_epochs``).
    :param max_epochs: Number of training epochs; controls how many epoch boundaries are flagged.
    :return: Dict mapping each artefact step index to its reason label string.
    """
    out: dict[int, str] = {0: "step0"}
    for n in range(1, max_epochs + 1):
        out.setdefault(n * steps_per_epoch - 1, "epoch_last_high")
        out.setdefault(n * steps_per_epoch, "epoch_first_low")
    return out


def mark_outliers(df_run: pd.DataFrame,
                  max_epochs: int = MAX_EPOCHS_SWEEP) -> pd.DataFrame:
    """Flag artefact and statistical outliers in ``samples_per_sec`` for one run.

    Two passes run in order:

    1. **Rule-based pass** — calls :func:`get_rule_outlier_steps` and flags step 0
       and every epoch-boundary step unconditionally, regardless of their value.
    2. **3xIQR pass** — applied only to rows that survived pass 1 and have a
       non-null ``samples_per_sec``. Uses a 3xIQR (vs. the typical 1.5x) to catch
       only genuine stragglers without discarding natural run-to-run spread.

    ``steps_per_epoch`` is derived as ``nunique(step) // max_epochs`` and stored in
    ``df.attrs["steps_per_epoch"]`` so downstream callers can inspect it.

    :param df_run: Tidy DataFrame for a single run, as produced by :func:`build_df_batch_sweep_run`.
    :param max_epochs: Epoch count used to locate epoch-boundary steps.
    :return: Copy of ``df_run`` with ``is_outlier`` and ``outlier_reason`` populated.
    """
    df = df_run.copy()
    if df.empty:
        return df
    steps_per_epoch = max(1, df["step"].nunique() // max_epochs)

    # First mark the known artefact steps based on their fixed positions relative to epoch boundaries.
    rule = get_rule_outlier_steps(steps_per_epoch, max_epochs)
    mask_rule = df["step"].isin(rule.keys())
    df.loc[mask_rule, "is_outlier"] = True
    df.loc[mask_rule, "outlier_reason"] = df.loc[mask_rule, "step"].map(rule).fillna("")

    # Then apply the 3xIQR rule to the remaining clean steps with valid throughput values, if there are enough of them.
    clean_mask = ~df["is_outlier"] & df["samples_per_sec"].notna()
    if clean_mask.sum() >= 8:
        q1, q3 = df.loc[clean_mask, "samples_per_sec"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        iqr_mask = clean_mask & ((df["samples_per_sec"] < lo) | (df["samples_per_sec"] > hi))
        df.loc[iqr_mask, "is_outlier"] = True
        df.loc[iqr_mask, "outlier_reason"] = "iqr"
    df.attrs["steps_per_epoch"] = steps_per_epoch
    return df


# ── Experiment-level loaders ───────────────────────────────────────────────────

def load_experiment(label: str,
                    db_path: Optional[Path],
                    batch_sizes: list[int] = EXPECTED_BATCH_SIZES,
                    max_epochs: int = MAX_EPOCHS_SWEEP,
                    BATCH_SWEEP_METRICS: list[str] = BATCH_SWEEP_METRICS,
                    batch_sweep_cols: list[str] = BATCH_SWEEP_COLS) -> pd.DataFrame:
    """Load all runs from one MLflow SQLite DB, build tidy dataframes for each run, and mark outliers.

    :param label: Experiment label written into every row's ``experiment`` column,
                  e.g. ``"no_aug"`` or ``"with_aug"``.
    :param db_path: Path to the MLflow SQLite tracking store; ``None`` or missing
                    path returns an empty frame with the correct columns.
    :param batch_sizes: Batch sizes to retain; runs with other sizes are ignored.
    :param max_epochs: Passed to :func:`mark_outliers` for epoch-boundary detection.
    :param BATCH_SWEEP_METRICS: MLflow metric keys to fetch per run.
    :param batch_sweep_cols: Output column schema.
    :return: Concatenated tidy DataFrame for all matching runs, or an empty frame.
    """
    if db_path is None or not db_path.exists():
        print(f"[WARN] DB not found for '{label}'. Returning empty frame.")
        return pd.DataFrame(columns=batch_sweep_cols)

    with closing(sqlite3.connect(str(db_path))) as con:
        run_batch_map: pd.DataFrame = fetch_run_batch_map(con, batch_sizes)
        runs_with_marked_outliers = [
            mark_outliers(build_df_batch_sweep_run(con,
                                                   run_batch_map_item["run_uuid"],
                                                   int(run_batch_map_item["batch_size"]),
                                                   label,
                                                   BATCH_SWEEP_METRICS,
                                                   batch_sweep_cols),
                          max_epochs)
            for _, run_batch_map_item in run_batch_map.iterrows()
        ]

    if not runs_with_marked_outliers:
        return pd.DataFrame(columns=batch_sweep_cols)
    return pd.concat(runs_with_marked_outliers, ignore_index=True)


def load_experiments(db_paths: dict[str, Path],
                     batch_sizes: list[int] = EXPECTED_BATCH_SIZES,
                     max_epochs: int = MAX_EPOCHS_SWEEP,
                     BATCH_SWEEP_METRICS: list[str] = BATCH_SWEEP_METRICS,
                     batch_sweep_cols: list[str] = BATCH_SWEEP_COLS) -> pd.DataFrame:
    """Load multiple experiments and concatenate them into one tidy DataFrame.

    :param db_paths: Ordered mapping of experiment label → MLflow SQLite path.
    :param batch_sizes: Batch sizes to retain across all experiments.
    :param max_epochs: Passed to :func:`mark_outliers` for epoch-boundary detection.
    :param BATCH_SWEEP_METRICS: MLflow metric keys to fetch per run.
    :param batch_sweep_cols: Output column schema.
    :return: Concatenated tidy DataFrame with an ``experiment`` column distinguishing sources.
    """
    frames = [
        load_experiment(label, path, batch_sizes, max_epochs, BATCH_SWEEP_METRICS, batch_sweep_cols)
        for label, path in db_paths.items()
    ]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} rows:")
    print(df.groupby(["experiment", "batch_size"]).size().rename("rows").to_frame())
    return df


# ── Throughput and GPU summaries ───────────────────────────────────────────────

def throughput_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ``samples_per_sec`` statistics per (experiment, batch_size).

    Outlier rows (``is_outlier=True``) are excluded before aggregation. Groups with
    zero clean ``samples_per_sec`` values are dropped.

    :param df: Concatenated tidy DataFrame from :func:`load_experiments` or :func:`load_experiment`.
    :return: One row per (experiment, batch_size) with columns
             ``[experiment, batch_size, p10, median, p90, median_time_per_step_ms,
             n_clean, n_outliers]``.
    """
    rows = []
    for (exp, bs), g in df.groupby(["experiment", "batch_size"]):
        n_outliers = int(g["is_outlier"].sum())
        clean = g[~g["is_outlier"]]
        vals = clean["samples_per_sec"].dropna().values
        if len(vals) == 0:
            continue
        rows.append({
            "experiment": exp,
            "batch_size": bs,
            "p10": float(np.percentile(vals, 10)),
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)),
            "median_time_per_step_ms": float(clean["time_per_step_ms"].median()),
            "n_clean": int(len(clean)),
            "n_outliers": n_outliers,
        })
    return pd.DataFrame(rows).sort_values(["experiment", "batch_size"]).reset_index(drop=True)


def gpu_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate GPU utilisation and memory statistics per (experiment, batch_size).

    Outlier rows are excluded before aggregation. GPU columns absent from older runs
    (all-NaN for a group) produce ``NaN`` instead of raising.

    :param df: Concatenated tidy DataFrame from :func:`load_experiments`.
    :return: One row per (experiment, batch_size) with columns
             ``[experiment, batch_size, median_gpu_util_pct, median_gpu_mem_gb,
             peak_gpu_mem_gb]``.
    """
    clean = df[~df["is_outlier"]]
    rows = []
    for (exp, bs), g in clean.groupby(["experiment", "batch_size"]):
        rows.append({
            "experiment": exp,
            "batch_size": bs,
            "median_gpu_util_pct": float(g["gpu_util_pct"].median()),
            "median_gpu_mem_gb": float(g["gpu_mem_gb"].median()),
            "peak_gpu_mem_gb": float(g["gpu_mem_gb"].max()),
        })
    return pd.DataFrame(rows).sort_values(["experiment", "batch_size"]).reset_index(drop=True)


# ── Placeholder generation ─────────────────────────────────────────────────────

def make_placeholder(label: str,
                     batch_sizes: list[int] = EXPECTED_BATCH_SIZES,
                     max_epochs: int = MAX_EPOCHS_SWEEP,
                     n_train: int = N_TRAIN_IMAGES_ISIC2017) -> pd.DataFrame:
    """Generate a synthetic batch-sweep DataFrame with outliers pre-marked.

    Useful for notebook development when the real MLflow DB is unavailable. Output
    is deterministic for identical arguments (fixed RNG seed).

    :param label: Experiment label written into the ``experiment`` column.
    :param batch_sizes: Batch sizes to simulate.
    :param max_epochs: Number of training epochs to simulate.
    :param n_train: Training set size; controls steps per epoch (``n_train // batch_size``).
    :return: Tidy DataFrame conforming to :data:`BATCH_SWEEP_COLS` with outliers flagged.
    """
    rng = np.random.default_rng(42)
    frames = []
    for bs in batch_sizes:
        steps_per_epoch = max(1, n_train // bs)
        total_steps = max_epochs * steps_per_epoch
        base_sps = 100.0 + 3.0 * bs
        rows = []
        for step in range(total_steps):
            rows.append({
                "experiment": label,
                "batch_size": bs,
                "run_uuid": f"placeholder-bs{bs}",
                "step": step,
                "samples_per_sec": float(rng.normal(base_sps, 4.0)),
                "time_per_step_ms": float(bs / base_sps * 1000.0),
                "epoch_idx": float(step // steps_per_epoch),
                "gpu_mem_gb": float(0.5 + bs * 0.05),
                "gpu_util_pct": float(min(99.0, 40.0 + bs * 0.5)),
                "train_loss": float(0.5 * np.exp(-step / 200.0)),
                "is_outlier": False,
                "outlier_reason": "",
            })
        df_run = pd.DataFrame(rows, columns=BATCH_SWEEP_COLS)
        frames.append(mark_outliers(df_run, max_epochs=max_epochs))
    if not frames:
        return pd.DataFrame(columns=BATCH_SWEEP_COLS)
    return pd.concat(frames, ignore_index=True)


# ── Scaling metrics ────────────────────────────────────────────────────────────

def add_scaling_metrics(summary: pd.DataFrame,
                        ref_bs: int) -> pd.DataFrame:
    """Add ``perfect_scaling`` and ``efficiency_pct`` columns to a throughput summary.

    Both columns are computed per experiment relative to ``ref_bs``. When ``ref_bs``
    is absent from an experiment's data, both columns are filled with ``NaN``.

    - ``perfect_scaling`` — ideal linear throughput at each batch size:
      ``(batch_size / ref_bs) * median_at_ref_bs``.
    - ``efficiency_pct`` — actual vs ideal: ``(median / perfect_scaling) * 100``.

    :param summary: Output of :func:`throughput_summary`.
    :param ref_bs: Reference batch size used as the baseline (100 % efficiency).
    :return: Copy of ``summary`` with ``perfect_scaling`` and ``efficiency_pct`` columns added.
    """
    out = summary.copy()
    out["perfect_scaling"] = np.nan
    out["efficiency_pct"] = np.nan
    for exp, g in out.groupby("experiment"):
        ref_rows = g[g["batch_size"] == ref_bs]
        if ref_rows.empty:
            continue
        ref_median = float(ref_rows["median"].iloc[0])
        idx = g.index
        out.loc[idx, "perfect_scaling"] = (out.loc[idx, "batch_size"] / ref_bs) * ref_median
        out.loc[idx, "efficiency_pct"] = (out.loc[idx, "median"] / out.loc[idx, "perfect_scaling"]) * 100.0
    return out


def add_max_efficiency(summary: pd.DataFrame) -> pd.DataFrame:
    """Add ``eff_max_pct`` — throughput as a percentage of the per-experiment peak.

    ``eff_max_pct = (median / max(median)) * 100`` within each experiment group.

    :param summary: Output of :func:`throughput_summary` (must have a ``median`` column).
    :return: Copy of ``summary`` with an ``eff_max_pct`` column added.
    """
    out = summary.copy()
    out["eff_max_pct"] = np.nan
    for exp, g in out.groupby("experiment"):
        peak = g["median"].max()
        if peak > 0:
            out.loc[g.index, "eff_max_pct"] = (g["median"] / peak) * 100.0
    return out


def plateau_batch_sizes(summary: pd.DataFrame,
                        threshold_pct: float = 80.0) -> list[int]:
    """Return batch sizes where every experiment exceeds ``threshold_pct`` of its peak.

    A batch size qualifies only when *all* experiments have ``eff_max_pct >= threshold_pct``
    for that batch size.

    :param summary: DataFrame with ``batch_size``, ``experiment``, and ``eff_max_pct`` columns.
    :param threshold_pct: Minimum efficiency percentage required in every experiment.
    :return: Sorted list of qualifying batch sizes.
    """
    passing = summary[summary["eff_max_pct"] >= threshold_pct]
    n_experiments = summary["experiment"].nunique()
    counts = passing.groupby("batch_size")["experiment"].nunique()
    qualified = counts[counts == n_experiments].index.tolist()
    return sorted(qualified)


# ── Recommendation ─────────────────────────────────────────────────────────────

def recommendation_facts(summary: pd.DataFrame,
                         gpu_tbl: pd.DataFrame,
                         recommended_bs: int,
                         threshold_pct: float = 80.0) -> dict:
    """Extract all numeric facts needed to render the notebook recommendation section.

    Centralises every number the conclusion cell needs so the prose template stays
    free of DataFrame lookups. The returned dict has the following top-level keys:

    - ``recommended_bs`` — the chosen batch size (int).
    - ``plateau_bs`` — list of batch sizes from :func:`plateau_batch_sizes` at ``threshold_pct``.
    - ``per_exp`` — dict keyed by experiment label, each entry containing
      ``peak_bs``, ``peak_median``, ``median_at_recommended``, and a
      ``table`` list of per-batch-size rows.
    - ``gpu_mem`` — dict of ``{batch_size: peak_gpu_mem_gb}`` across all experiments.
    - ``gpu_util_at_recommended`` — dict of ``{experiment: median_gpu_util_pct}``
      at the recommended batch size.
    - ``aug_penalty_pct_at_recommended`` — percentage throughput drop from ``no_aug``
      to ``with_aug`` at the recommended batch size; present only when both experiments exist.

    :param summary: Output of :func:`throughput_summary` (must have a ``median`` column).
    :param gpu_tbl: Output of :func:`gpu_summary`.
    :param recommended_bs: The chosen batch size, set as a human judgment call in the notebook config.
    :param threshold_pct: Efficiency threshold forwarded to :func:`plateau_batch_sizes`.
    :return: Nested dict of facts as described above.
    """
    facts: dict = {"recommended_bs": int(recommended_bs)}
    if "eff_max_pct" in summary.columns:
        facts["plateau_bs"] = plateau_batch_sizes(summary, threshold_pct=threshold_pct)
    else:
        facts["plateau_bs"] = []
    facts["per_exp"] = {}
    for exp, g in summary.groupby("experiment"):
        g = g.sort_values("batch_size")
        peak_row = g.loc[g["median"].idxmax()]
        at_rec = g[g["batch_size"] == recommended_bs]
        facts["per_exp"][exp] = {
            "peak_bs": int(peak_row["batch_size"]),
            "peak_median": round(float(peak_row["median"]), 2),
            "median_at_recommended": round(float(at_rec["median"].iloc[0]), 2)
            if not at_rec.empty else None,
            "table": g[["batch_size", "median", "median_time_per_step_ms"]].round(2).to_dict(orient="records"),
        }
    facts["gpu_mem"] = (
        gpu_tbl.groupby("batch_size")["peak_gpu_mem_gb"].max().round(2).to_dict()
    )
    facts["gpu_util_at_recommended"] = {}
    for exp, g in gpu_tbl.groupby("experiment"):
        sub = g[g["batch_size"] == recommended_bs]
        facts["gpu_util_at_recommended"][exp] = (
            round(float(sub["median_gpu_util_pct"].iloc[0]), 1)
            if not sub.empty and pd.notna(sub["median_gpu_util_pct"].iloc[0])
            else None
        )
    if {"no_aug", "with_aug"}.issubset(summary["experiment"].unique()):
        med_no = facts["per_exp"].get("no_aug", {}).get("median_at_recommended")
        med_aug = facts["per_exp"].get("with_aug", {}).get("median_at_recommended")
        if med_no and med_aug:
            facts["aug_penalty_pct_at_recommended"] = round(
                100.0 * (1.0 - med_aug / med_no), 2
            )
    return facts
