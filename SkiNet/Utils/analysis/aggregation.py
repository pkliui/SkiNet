"""Aggregation helpers that build summary DataFrames from raw MLflow tables.

Column names used throughout this module (``key``, ``value``, ``step``,
``timestamp``, ``run_uuid``) come directly from MLflow's SQLite database
schema and are preserved as-is after loading.  The two tables consumed most
often are:

- ``latest_metrics`` — one row per (run, metric), holding only the **most
  recently logged** value.  Columns: ``run_uuid``, ``key``, ``value``,
  ``step``, ``timestamp``, ``is_nan``.
- ``metrics`` — one row per (run, metric, step), the **full epoch history**.
  Same columns as ``latest_metrics``.

So when the code filters ``df[df["key"].eq("final/val_dice")]`` and reads
``df["value"]``, it is querying those MLflow column names directly.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd

from SkiNet.Utils.analysis.io import load_mlflow_tables
from SkiNet.Utils.analysis.parsing import parse_encoder_merge
from SkiNet.Utils.analysis.schema import ARCH, SEED


# ---------------------------------------------------------------------------
# Pre-flight inventories — audit what was logged before summarising
# ---------------------------------------------------------------------------


def metric_inventory(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return one row per metric key summarising its coverage across all runs.

    Diagnostic/audit helper — use this before calling ``summarize_runs`` to
    verify which metrics were actually logged, how many runs recorded them, how
    many data points in total, and what step and value ranges look like.

    ``summarize_runs`` is the primary analysis table that all other functions
    consume; this function is a pre-flight check to confirm the metrics you
    plan to pass to it actually exist in the data.

    Example output::

        # key                       runs  points  min_step  max_step  min_value  max_value
        # val_best_dice_at_thr...   12    600     0         49        0.41       0.89
        # val_dice                  12    600     0         49        0.38       0.87
        # system/gpu_utilization    12    600     0         49        42.1       99.8

    :param metrics: Raw MLflow metrics table (one row per run/key/step data point).
    :return: DataFrame with columns [key, runs, points, min_step, max_step,
             min_value, max_value], sorted alphabetically by key.
    """
    return (
        metrics.groupby("key")          # one group per metric name
        .agg(
            runs=("run_uuid", "nunique"),   # distinct runs that logged this metric
            points=("value", "size"),       # total data points across all runs and steps
            min_step=("step", "min"),       # earliest step at which it was logged
            max_step=("step", "max"),       # latest step at which it was logged
            min_value=("value", "min"),     # lowest recorded value
            max_value=("value", "max"),     # highest recorded value
        )
        .reset_index()          # promote "key" from index back to a column
        .sort_values("key")     # alphabetical order for deterministic output
    )


def parameter_inventory(params: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split logged hyperparameters into constant and variable inventories.

    Diagnostic/audit helper — use this before calling ``summarize_runs`` to
    verify which hyperparameters were actually logged, how many runs recorded
    them, how many data points in total, and what value ranges look like.

    Pivots the long-format params table into a wide matrix (runs x params), then
    classifies each param by how many distinct values it takes across runs.
    Constants (n_values=1, same everywhere) are split from sweep variables
    (n_values>1, your actual experiment dimensions).

    Example — given params logged across 4 runs::

        # Input (long format, one row per run/param):
        # run_uuid  key         value
        # aaa...    loss_fn     dice
        # aaa...    encoder     resnet50
        # bbb...    loss_fn     dice
        # bbb...    encoder     resnet34

        constants, variables = parameter_inventory(params)

        # constants (n_values == 1):
        # param    n_values  values
        # loss_fn  1         dice

        # variables (n_values > 1):
        # param    n_values  values
        # encoder  2         resnet34, resnet50

    :param params: Raw MLflow params table (one row per run/key/value).
    :return: Tuple of (constants_df, variables_df). Each has columns
             [param, n_values, values] where ``values`` is a preview string
             of up to 4 distinct values separated by commas.
    """
    # reshape long (run, key, value) → wide matrix: rows=runs, cols=param names
    # aggfunc="first" collapses duplicate entries (MLflow can log the same param twice)
    wide = params.pivot_table(index="run_uuid", columns="key", values="value", aggfunc="first")

    rows = []
    for key in wide.columns:
        # drop runs that never logged this param, normalise to str so numeric
        # variants ("0.001" vs 0.001) don't inflate the unique count
        values = sorted(wide[key].dropna().astype(str).unique())
        rows.append(
            {
                "param": key,
                "n_values": len(values),
                # preview string: up to 4 values; truncated with " ..." if more
                "values": ", ".join(values[:4]) + (" ..." if len(values) > 4 else ""),
            }
        )

    # sort so n_values=1 (constants) rise to the top; alphabetical within each tier
    inventory = pd.DataFrame(rows).sort_values(["n_values", "param"])

    # constants: same value in every run — not a sweep dimension
    const: pd.DataFrame = inventory[inventory["n_values"].eq(1)]
    # variables: differ across runs — these are the actual experiment axes
    variable: pd.DataFrame = inventory[inventory["n_values"].gt(1)]
    return const, variable


# ---------------------------------------------------------------------------
# Per-run summary — the core table every view below consumes
# ---------------------------------------------------------------------------

def summarize_runs(tables: dict[str, pd.DataFrame],
                   monitor: str,
                   tail_n: int = 10,
                   sort_by: str = "best") -> pd.DataFrame:
    """Build one production-oriented summary row per MLflow run - the primary analysis table.

    Combines metadata, final-epoch scalars, best-epoch statistics, and tail
    stability statistics into a single wide DataFrame. This is the primary
    analysis table — most other functions consume its output.

    Final metrics (``final/`` prefix) are read from the ``latest`` table via a
    direct O(1) lookup since they are logged only once at the end of training.
    Per-epoch metrics (max, min and tail mean) are read from the full ``metrics`` history to find best
    epochs and tail stability.

    :param tables: Output of ``load_mlflow_tables``
    :param monitor: Validation metric used for peak/tail columns and primary sort key. E.g. "val_dice"
    :param tail_n: Number of final epochs over which to compute tail mean and std stability columns. Defaults to 10.
    :param sort_by: ``"best"`` to sort by ``{monitor}_max`` (peak epoch), ``"tail"`` to sort by
        ``{monitor}_tail_mean`` (tail stability). Defaults to ``"best"``.
    :return: DataFrame with one row per run, sorted descending by the chosen column.
    """

    # Extract tables
    runs = tables["runs"]
    metrics = tables["metrics"]
    latest = tables["latest"]
    records = []

    # Iterate over runs and extract metrics
    for _, run in runs.iterrows():
        # get the current run uuid and filter the metrics and latest tables for this run uuid
        run_uuid = run["run_uuid"]
        run_latest: pd.DataFrame = latest[latest["run_uuid"].eq(run_uuid)]
        run_metrics: pd.DataFrame = metrics[metrics["run_uuid"].eq(run_uuid)]
        # get the encoder and merge modes from the experiment name
        encoder, merge = parse_encoder_merge(str(run["experiment_name"]))

        # pick run-specific attributes and the latest value for each of the final metrics
        record = {
            "experiment_id": int(run["experiment_id"]),
            "run_uuid": run_uuid,
            "run_short": run_uuid[:8],
            "status": run["status"],
            "encoder": encoder,
            "merge": merge,
            "duration_min": (float(run["end_time"]) - float(run["start_time"])) / 60_000,
            "final_train_dice": _latest_value(run_latest, "final/train_dice"),
            "final_val_dice": _latest_value(run_latest, "final/val_dice"),
            "final_val_iou": _latest_value(run_latest, "final/val_iou"),
            "final_val_loss": _latest_value(run_latest, "final/val_loss"),
            "samples_per_sec": _latest_value(run_latest, "final/perf/samples_per_sec"),
            "time_per_step_ms": _latest_value(run_latest, "final/perf/time_per_step_ms"),
            "final_grad_scale": _latest_value(run_latest, "final/grad_scale"),
        }

        # Peak and tail stability for each metric of interest.
        # Column naming convention: {key}_max / {key}_min for the peak,
        # {key}_tail_mean / {key}_tail_std for the tail window.
        for key, mode in (
            (monitor, "max"),
            ("val_dice", "max"),
            ("val_iou", "max"),
            ("val_loss", "min"),
        ):
            # merge the returned dicts into the record for this run
            record.update(_best_metric_columns(run_metrics, key, mode))
            record.update(_tail_metric_columns(run_metrics, key, tail_n))

        # add generalization gap and drops from peak to final and tail
        record["generalization_gap_final"] = record["final_train_dice"] - record["final_val_dice"]
        record["drop_peak_to_final"] = record[f"{monitor}_max"] - record.get(f"final_{monitor}", float("nan"))
        record["gap_peak_to_tail"] = record[f"{monitor}_max"] - record.get(f"{monitor}_tail_mean", float("nan"))
        records.append(record)

    if sort_by not in ("best", "tail"):
        raise ValueError(f"sort_by must be 'best' or 'tail', got {sort_by!r}")
    df = pd.DataFrame(records)
    if df.empty:
        return df
    sort_col = f"{monitor}_max" if sort_by == "best" else f"{monitor}_tail_mean"
    return df.sort_values(sort_col, ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Single-experiment views — consume one summarize_runs output
# ---------------------------------------------------------------------------

def rank_runs(
    run_summary: pd.DataFrame,
    monitor: str,
    sort_by: str = "tail",
    tail_n: int = 10,
) -> "pd.io.formats.style.Styler":
    """Return a styled ranking table sorted by a monitor-derived column.

    Column selection and format strings are derived from ``monitor`` so the
    same function works for any checkpoint metric without manual edits.
    Columns that are absent from ``run_summary`` (e.g. no artifact root) are
    silently dropped rather than raising a KeyError.

    :param run_summary: Output of ``summarize_runs``.
    :param monitor: Validation metric used as the primary sort key.
    :param sort_by: ``"best"`` to rank by the peak epoch value
                    (``best_{monitor}``), or ``"tail"`` to rank by the tail
                    mean over the last ``tail_n`` epochs
                    (``best_{monitor}_tail_mean``).
    :param tail_n: Number of final epochs in the tail window — used only in
                   the caption when ``sort_by="tail"``. Must match the value
                   passed to ``summarize_runs``.
    :return: Styled DataFrame ready for Jupyter display.
    """
    if sort_by not in ("best", "tail"):
        raise ValueError(f"sort_by must be 'best' or 'tail', got {sort_by!r}")

    sort_col = f"{monitor}_max" if sort_by == "best" else f"{monitor}_tail_mean"
    sort_label = (
        "peak value across all epochs"
        if sort_by == "best"
        else f"tail mean over last {tail_n} epochs"
    )
    cols = [
        "encoder", "merge", "run_short", "status",
        f"{monitor}_max", f"{monitor}_tail_mean", f"{monitor}_tail_std",
        f"{monitor}_max_epoch",
        "val_dice_max", "val_iou_max", "val_loss_min",
        f"final_{monitor}", "final_val_dice",
        "drop_peak_to_final", "gap_peak_to_tail",
        "generalization_gap_final", "samples_per_sec", "duration_min",
    ]
    present = list(dict.fromkeys(c for c in cols if c in run_summary.columns))
    fmt = {
        f"{monitor}_max": "{:.4f}",
        f"{monitor}_tail_mean": "{:.4f}",
        f"{monitor}_tail_std": "{:.4f}",
        "val_dice_max": "{:.4f}",
        "val_iou_max": "{:.4f}",
        "val_loss_min": "{:.4f}",
        f"final_{monitor}": "{:.4f}",
        "final_val_dice": "{:.4f}",
        "drop_peak_to_final": "{:.4f}",
        "gap_peak_to_tail": "{:.4f}",
        "generalization_gap_final": "{:.4f}",
        "samples_per_sec": "{:.1f}",
        "duration_min": "{:.1f}",
    }
    caption = f"Run ranking — sorted by {sort_col} ({sort_label}, descending)"
    df = (
        run_summary[present].sort_values(sort_col, ascending=False)
        if sort_col in present else run_summary[present]
    ).reset_index(drop=True)

    best_col = f"{monitor}_max"
    tail_col = f"{monitor}_tail_mean"
    gradient_cols = [c for c in (tail_col, best_col) if c in df.columns]

    return (
        df.style
        .format({k: v for k, v in fmt.items() if k in df.columns})
        .background_gradient(subset=gradient_cols, cmap="RdYlGn")
        .set_caption(caption)
    )


def summarize_by_family(run_summary: pd.DataFrame, monitor: str, group_cols: Iterable[str] = ("encoder", "merge")) -> pd.DataFrame:
    """Aggregate accuracy, throughput, and generalization by architecture family.

    Each ``group_col`` is treated as an independent sweep dimension. For each
    dimension the runs are split by that column's value and all other dimensions
    are marginalised out. In a 3-encoder × 3-merge sweep (9 runs total) this
    produces two blocks of three rows each:

    - ``encoder`` block: ``classical`` row averages the 3 runs that used the
      classical encoder (one per merge mode); ``he2`` and ``se`` rows do the
      same. Each row answers "ignoring merge choice, how good is this encoder?"
    - ``merge`` block: symmetrically averages over encoder variants for each
      merge mode. Answers "ignoring encoder choice, how good is this merge?"

    All ``mean_*``, ``std_*``, and ``max_*`` columns aggregate **across the
    runs in that group** (not across epochs within a run).

    The output is sorted by ``family`` ascending then ``mean_best_dice``
    descending, so within each family the strongest architecture appears first.

    Example output for a 3×3 sweep (encoder and merge blocks stacked)::

        # family   value       n  mean_best_dice  max_best_dice  mean_final_dice  mean_samples_per_sec  mean_gap
        # encoder  resnet50    3  0.861           0.889          0.854            48.2                  0.028
        # encoder  resnet34    3  0.843           0.871          0.836            61.7                  0.031
        # merge    attention   3  0.855           0.889          0.848            44.1                  0.029
        # merge    add         3  0.849           0.872          0.842            65.8                  0.030

    Column guide (all aggregated across runs within the group):
    - ``n`` — number of runs in this group
    - ``mean_best_dice`` — mean of each run's single peak-epoch Dice; central
      tendency of how good this family gets at its best
    - ``max_best_dice`` — highest single-run peak; the ceiling this family
      achieved in at least one configuration
    - ``mean_final_dice`` — mean of each run's last-epoch Dice (not best epoch);
      reflects whether runs stay near their peak or degrade after it
    - ``mean_samples_per_sec`` — mean throughput; not in the heatmap, keep for
      cost/speed comparison across families
    - ``mean_gap`` — mean generalization gap (train − val dice); not in the
      heatmap, key signal for overfitting tendency by family

    :param run_summary: Output of ``summarize_runs``.
    :param group_cols: Columns to group by independently. Defaults to
                       ``("encoder", "merge")``.
    :return: Styled DataFrame with columns [family, value, n, mean_best_dice,
             max_best_dice, mean_final_dice, mean_samples_per_sec, mean_gap].
    """
    summaries = []
    for group_col in group_cols:
        max_col = f"{monitor}_max"
        final_col = f"final_{monitor}"
        grouped = (
            run_summary.groupby(group_col)
            .agg(
                n=("run_uuid", "count"),
                mean_best_dice=(max_col, "mean"),
                max_best_dice=(max_col, "max"),
                mean_tail_mean=(f"{monitor}_tail_mean", "mean"),
                mean_final_dice=(final_col, "mean") if final_col in run_summary.columns else ("run_uuid", "count"),
                mean_samples_per_sec=("samples_per_sec", "mean"),
                mean_gap=("generalization_gap_final", "mean"),
            )
            .assign(family=group_col)
            .reset_index(names="value")
        )
        if f"final_{monitor}" not in run_summary.columns:
            grouped = grouped.drop(columns=["mean_final_dice"], errors="ignore")
        summaries.append(grouped)
    df = (
        pd.concat(summaries, ignore_index=True)
        .sort_values(["family", "mean_best_dice"], ascending=[True, False])
        .reset_index(drop=True)   # contiguous index required by Styler.background_gradient
    )
    # Put family first, then value, then the rest
    other_cols = [c for c in df.columns if c not in ("family", "value")]
    df = df[["family", "value"] + other_cols]

    fmt = {
        'mean_best_dice': '{:.4f}',
        'max_best_dice': '{:.4f}',
        'mean_tail_mean': '{:.4f}',
        'mean_final_dice': '{:.4f}',
        'mean_samples_per_sec': '{:.1f}',
        'mean_gap': '{:.4f}',
    }
    gradient_cols = [c for c in ('mean_tail_mean', 'mean_best_dice') if c in df.columns]

    # Each family gets its own gradient call so colours are normalised within
    # each block independently (encoder rows ranked against each other,
    # merge rows ranked against each other) — same RdYlGn scale throughout.
    families = list(dict.fromkeys(df["family"]))  # stable insertion order
    style = df.style.format({k: v for k, v in fmt.items() if k in df.columns})
    for fam in families:
        idx = df.index[df["family"] == fam].tolist()
        if not idx:
            continue
        for col in gradient_cols:
            col_vals = df.loc[idx, col].dropna()
            if col_vals.empty:
                continue
            style = style.background_gradient(
                subset=pd.IndexSlice[idx, [col]],
                cmap="RdYlGn",
                vmin=float(col_vals.min()),
                vmax=float(col_vals.max()),
            )
    return style


def epoch_metrics(metrics: pd.DataFrame, run_summary: pd.DataFrame, keys: Iterable[str]) -> pd.DataFrame:
    """Return epoch-indexed metric history joined with architecture labels.

    Filters the raw metric log to the requested keys, attaches encoder and merge
    labels from the run summary, and assigns a 1-based epoch number per
    run/metric combination. The result is a long-format DataFrame ready for
    plotting training curves grouped or coloured by architecture.

    Example::

        frame = epoch_metrics(metrics, run_summary, keys=["val_dice", "val_loss"])
        # run_uuid  key       value  step  encoder   merge      epoch  architecture
        # aaa...    val_dice  0.71   0     resnet50  attention  1      resnet50 + attention
        # aaa...    val_dice  0.79   1     resnet50  attention  2      resnet50 + attention
        # aaa...    val_loss  0.43   0     resnet50  attention  1      resnet50 + attention

    :param metrics: Raw MLflow metrics table.
    :param run_summary: Output of ``summarize_runs`` (provides encoder/merge labels).
    :param keys: Metric keys to include (e.g. ``["val_dice", "val_loss"]``).
    :return: Long-format DataFrame with added columns ``epoch`` (1-based) and
             ``architecture`` (``"{encoder} + {merge}"``).
    """
    frame = metrics[metrics["key"].isin(keys)].copy()
    frame = frame.merge(run_summary[["run_uuid", "encoder", "merge"]], on="run_uuid", how="left")
    frame = frame.sort_values(["run_uuid", "key", "step", "timestamp"])
    frame["epoch"] = frame.groupby(["run_uuid", "key"]).cumcount() + 1
    frame["architecture"] = frame["encoder"] + " + " + frame["merge"]
    return frame


def resource_summary(metrics: pd.DataFrame, run_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize peak system resource metrics per architecture.

    Extracts all metrics whose key starts with ``system/`` (e.g. GPU
    utilisation, GPU memory, CPU usage), attaches architecture labels, then
    pivots to a wide table where each ``system/`` key becomes a column
    containing its peak (max) observed value across the run. Useful for
    spotting resource bottlenecks and comparing hardware demands across
    encoder families.

    Example output::

        # architecture          system/gpu_mem_gb  system/gpu_utilization
        # resnet34 + add        10.2               87.4
        # resnet50 + attention  14.7               94.1

    :param metrics: Raw MLflow metrics table.
    :param run_summary: Output of ``summarize_runs`` (provides encoder/merge labels).
    :return: Wide DataFrame with one row per architecture and one column per
             ``system/`` metric key, values being the peak observed value.
             Sorted alphabetically by architecture.
    """
    system = metrics[metrics["key"].str.startswith("system/")].copy()
    system = system.merge(run_summary[["run_uuid", "encoder", "merge"]], on="run_uuid", how="left")
    system["architecture"] = system["encoder"] + " + " + system["merge"]
    return (
        system.groupby(["architecture", "key"])["value"]
        .agg(["mean", "max"])
        .reset_index()
        .pivot(index="architecture", columns="key", values="max")
        .reset_index()
        .sort_values("architecture")
    )


# ---------------------------------------------------------------------------
# Private helpers — column builders used by summarize_runs
# ---------------------------------------------------------------------------

def _latest_value(latest: pd.DataFrame, key: str) -> float:
    """Look up a single scalar from the MLflow latest_metrics table.

    The latest_metrics table holds only the most recent logged value per
    run/key, making this an O(1) lookup. Used for ``final/`` metrics which are
    logged only once at the end of training, so latest == final == only value.
    Returns NaN if the key was never logged for this run.

    :param latest: MLflow ``latest_metrics`` table pre-filtered to a single run.
        Columns ``key`` and ``value`` are MLflow's own schema names (see module docstring).
    :param key: Metric key to look up (e.g. ``"final/val_dice"``).
    :return: Scalar float value, or ``math.nan`` if not found.
    """
    sub = latest.loc[latest["key"].eq(key), "value"]
    return float(sub.iloc[0]) if len(sub) else math.nan


def _best_metric_columns(metrics: pd.DataFrame, key: str, mode: str) -> dict[str, float | int]:
    """Find the epoch with the best value for a metric and return it with its position.

    Output keys are ``{key}_{mode}``, ``{key}_{mode}_epoch``, ``{key}_{mode}_step``.
    Example: ``key="val_dice", mode="max"`` → ``val_dice_max``, ``val_dice_max_epoch``, ``val_dice_max_step``.

    :param metrics: Raw MLflow metrics table pre-filtered to a single run.
    :param key: Metric key to search (e.g. ``"val_dice"``).
    :param mode: ``"max"`` to find the highest value, ``"min"`` for the lowest.
    :return: Dict with keys ``{key}_{mode}``, ``{key}_{mode}_epoch``, ``{key}_{mode}_step``.
             All three are NaN if the metric was never logged for this run.
    """
    if mode not in ("max", "min"):
        raise ValueError(f"mode must be 'max' or 'min', got {mode!r} (key={key!r})")
    label = f"{key}_{mode}"
    metrics_at_specified_key = metrics.loc[metrics["key"].eq(key)].sort_values(["step", "timestamp"]).copy()
    if metrics_at_specified_key.empty:
        return {label: math.nan, f"{label}_epoch": math.nan, f"{label}_step": math.nan}
    metrics_at_specified_key["epoch"] = range(1, len(metrics_at_specified_key) + 1)
    idx = metrics_at_specified_key["value"].idxmax() if mode == "max" else metrics_at_specified_key["value"].idxmin()
    row = metrics_at_specified_key.loc[idx]
    return {label: float(row["value"]), f"{label}_epoch": int(row["epoch"]), f"{label}_step": int(row["step"])}


def _tail_metric_columns(metrics: pd.DataFrame, key: str, n: int) -> dict[str, float]:
    """Compute mean and std over the last n epochs for a metric.

    Output keys are ``{key}_tail_mean`` and ``{key}_tail_std``.

    :param metrics: Raw MLflow metrics table pre-filtered to a single run.
    :param key: Metric key to summarise (e.g. ``"val_dice"``).
    :param n: Number of final epochs to include in the window.
    :return: Dict with keys ``{key}_tail_mean`` and ``{key}_tail_std``.
             Both are NaN if the metric was never logged for this run.
    """
    metrics_at_specified_key = metrics.loc[metrics["key"].eq(key)].sort_values(["step", "timestamp"])
    tail = metrics_at_specified_key.tail(n)["value"]
    if tail.empty:
        return {f"{key}_tail_mean": math.nan, f"{key}_tail_std": math.nan}
    return {f"{key}_tail_mean": float(tail.mean()), f"{key}_tail_std": float(tail.std())}


# ---------------------------------------------------------------------------
# Multi-seed loader — paired-architecture runs across seeds (E2 tiebreak)
# ---------------------------------------------------------------------------

def load_runs(*dbs: Path,
              exp_map: dict[int, str],
              monitor: str,
              tail_n: int = 10) -> pd.DataFrame:
    """Load paired-architecture runs from MLflow DBs and complement them
    with peak and tail stability for each metric of interest,
    in addition to other summary statistics, as well as seed and architecture labels.

    All databases are concatenated and validated for balance
    (equal run counts per architecture).

    :param dbs: One or more paths to MLflow SQLite databases.
    :param exp_map: Mapping of MLflow ``experiment_id`` to architecture label,
         e.g. ``{1: "resnet", 2: "se", 3: "he2"}``
    :param monitor: Base metric forwarded to ``summarize_runs``.
    :param tail_n: Tail-epoch window forwarded to ``summarize_runs``. Default 10.
    :return: One row per (seed, architecture) with summary statistics plus
             ``seed`` and ``arch`` columns. Prints run count and seed list.
    :raises ValueError: If architectures have different run counts (unbalanced design).
    """
    def _one(db: Path) -> pd.DataFrame:
        tables = load_mlflow_tables(db)
        all_runs = summarize_runs(tables, monitor=monitor, tail_n=tail_n)
        additional_cols = tables["runs"][["run_uuid", "run_name", "experiment_id"]].assign(
            **{
                SEED: lambda d: d["run_name"].str.extract(r"seed(\d+)")[0].astype(int),
                ARCH: lambda d: d["experiment_id"].map(exp_map),
            }
        )
        return all_runs.merge(additional_cols[["run_uuid", SEED, ARCH]], on="run_uuid")

    df = pd.concat([_one(db) for db in dbs], ignore_index=True)
    n_seeds = df[SEED].nunique()
    counts = df[ARCH].value_counts()
    if not (counts == n_seeds).all():
        offenders = counts[counts != n_seeds].to_dict()
        raise ValueError(
            f"Unbalanced seed coverage: expected every architecture to have {n_seeds} seed(s) "
            f"({sorted(df[SEED].unique())}), "
            f"but these architectures differ: {offenders}. "
            f"Full breakdown: {counts.to_dict()}"
        )
    print(
        f"Loaded {len(df)} runs: {dict(df[ARCH].value_counts())} "
        f"| seeds: {sorted(df[SEED].unique())}"
    )
    return df
