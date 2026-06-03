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


def summarize_runs(tables: dict[str, pd.DataFrame],
                   artifact_root: Path | None = None,
                   monitor: str = "val_best_dice_at_threshold",
                   tail_n: int = 10) -> pd.DataFrame:
    """Build one production-oriented summary row per MLflow run - the primary analysis table.

    Combines metadata, final-epoch scalars, best-epoch statistics, and tail
    stability statistics into a single wide DataFrame. This is the primary
    analysis table — most other functions consume its output.

    Final metrics (``final/`` prefix) are read from the ``latest`` table via a
    direct O(1) lookup since they are logged only once at the end of training.
    Per-epoch metrics are read from the full ``metrics`` history to find best
    epochs and tail stability.

    Columns produced per run:

    - Identity: ``experiment_id``, ``run_uuid``, ``run_short`` (first 8 chars), ``status``
    - Architecture: ``encoder``, ``merge`` (parsed from experiment name)
    - Timing: ``duration_min``
    - Final-epoch scalars: ``final_train_dice``, ``final_val_dice``,
      ``final_val_iou``, ``final_val_loss``,
      ``samples_per_sec``, ``time_per_step_ms``, ``final_grad_scale``
    - Peak per metric: ``{monitor}_max``, ``val_dice_max``, ``val_iou_max``,
      ``val_loss_min`` — each with ``_epoch`` and ``_step`` siblings
    - Tail stability per metric: ``{monitor}_tail_mean``, ``{monitor}_tail_std``
      over the last ``tail_n`` epochs
    - ``generalization_gap_final`` — final_train_dice minus final_val_dice
    - Checkpoint: ``checkpoint``, ``checkpoint_mb`` (only if ``artifact_root`` given)
    - ``model_total_params`` (only if ``artifact_root`` given)

    Example output (one row per run, sorted by best monitor descending)::

        # run_short  encoder   merge      duration_min  final_val_dice
        # a3f2b1c0  resnet50  attention  187.4         0.871
        # 9d1e4a77  resnet50  add        192.1         0.858
        # c7b09f23  resnet34  attention  141.2         0.847
        # 5e2a1d88  resnet34  add        138.8         0.831

        # Reading the table:
        # - val_dice_max       → peak Dice across all epochs (ceiling)
        # - val_dice_max_epoch → which epoch that peak occurred at
        # - val_dice_tail_mean → mean Dice over last 10 epochs (stability)
        # - val_dice_tail_std  → std over last 10 epochs (noise level)
        # - generalization_gap_final → train_dice - val_dice at final epoch (overfitting signal)

    :param tables: Output of ``load_mlflow_tables`` — dict with keys
                   ``"runs"``, ``"metrics"``, ``"latest"``.
    :param artifact_root: Optional MLflow artifact directory root. When given,
                          checkpoint info and model param count are added.
    :param monitor: Validation metric used for checkpoint selection and primary
                    sort key. Defaults to ``"val_dice"``.
    :param tail_n: Number of final epochs over which to compute tail mean and
                   std stability columns. Defaults to 10.
    :return: DataFrame with one row per run, sorted by ``{monitor}_max``
             descending (best run first).
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

        # pick the latest value for each of the final metrics from latest_metrics (direct lookup O(1)),
        # and the best value for the monitor metric
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
        for key, suffix, mode in (
            (monitor, f"{monitor}_max", "max"),
            ("val_dice", "val_dice_max", "max"),
            ("val_iou", "val_iou_max", "max"),
            ("val_loss", "val_loss_min", "min"),
        ):
            record.update(_best_metric_columns(run_metrics, key, suffix, mode))
            record.update(_tail_metric_columns(run_metrics, key, key, tail_n))

        record["generalization_gap_final"] = record["final_train_dice"] - record["final_val_dice"]
        record["drop_peak_to_final"] = record[f"{monitor}_max"] - record.get(f"final_{monitor}", float("nan"))
        record["gap_peak_to_tail"] = record[f"{monitor}_max"] - record.get(f"{monitor}_tail_mean", float("nan"))

        if artifact_root is not None:
            record.update(_checkpoint_info(artifact_root, run_uuid, int(run["experiment_id"])))
            record["model_total_params"] = _model_total_params(artifact_root, run_uuid, int(run["experiment_id"]))
        records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.sort_values(f"{monitor}_max", ascending=False).reset_index(drop=True)


def rank_table(
    run_summary: pd.DataFrame,
    monitor: str = "val_dice",
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
        "checkpoint", "checkpoint_mb", "model_total_params",
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
        "checkpoint_mb": "{:.1f}",
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


def family_summary(run_summary: pd.DataFrame, group_cols: Iterable[str] = ("encoder", "merge"), monitor: str = "val_dice") -> pd.DataFrame:
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


def _best_metric_columns(metrics: pd.DataFrame, key: str, label: str, mode: str) -> dict[str, float | int]:
    """Find the epoch with the best value for a metric and return it with its position.

    Scans the full epoch history for ``key`` and picks the row with the maximum
    (mode="max") or minimum (mode="min") value. Returns three columns: the best
    value itself, the 1-based epoch number, and the raw MLflow step at which it
    occurred. All three are NaN if the metric was never logged for this run.

    Example — for a run with 50 epochs of val_dice::

        _best_metric_columns(run_metrics, "val_dice", "best_val_dice", "max")
        # {"best_val_dice": 0.887, "best_val_dice_epoch": 38, "best_val_dice_step": 37}

    :param metrics: Raw MLflow metrics table pre-filtered to a single run.
    :param key: Metric key to search (e.g. ``"val_dice"``).
    :param label: Output column name prefix (e.g. ``"best_val_dice"``).
    :param mode: ``"max"`` to find the highest value, ``"min"`` for the lowest.
    :return: Dict with keys ``{label}``, ``{label}_epoch``, ``{label}_step``.
    """
    sub = metrics.loc[metrics["key"].eq(key)].sort_values(["step", "timestamp"]).copy()
    if sub.empty:
        return {label: math.nan, f"{label}_epoch": math.nan, f"{label}_step": math.nan}
    sub["epoch"] = range(1, len(sub) + 1)
    idx = sub["value"].idxmax() if mode == "max" else sub["value"].idxmin()
    row = sub.loc[idx]
    return {label: float(row["value"]), f"{label}_epoch": int(row["epoch"]), f"{label}_step": int(row["step"])}


def _tail_metric_columns(metrics: pd.DataFrame, key: str, label: str, n: int) -> dict[str, float]:
    """Compute mean and std over the last n epochs for a metric.

    Complements ``_best_metric_columns`` by measuring convergence stability
    rather than peak performance. A run where tail_mean ≈ best value is stable
    and has converged; a large gap between them means the model peaked early
    then degraded or oscillated. The std quantifies residual epoch-to-epoch
    noise in the final training window.

    Example — a run that peaked at epoch 30 then degraded::

        _tail_metric_columns(run_metrics, "val_dice", "best_val_dice", n=10)
        # {"best_val_dice_tail_mean": 0.841, "best_val_dice_tail_std": 0.012}
        # vs best_val_dice = 0.887 — large gap signals instability

    :param metrics: Raw MLflow metrics table pre-filtered to a single run.
    :param key: Metric key to summarise (e.g. ``"val_dice"``).
    :param label: Output column name prefix (e.g. ``"best_val_dice"``).
    :param n: Number of final epochs to include in the window.
    :return: Dict with keys ``{label}_tail_mean`` and ``{label}_tail_std``.
             Both are NaN if the metric was never logged for this run.
    """
    sub = metrics.loc[metrics["key"].eq(key)].sort_values(["step", "timestamp"])
    tail = sub.tail(n)["value"]
    if tail.empty:
        return {f"{label}_tail_mean": math.nan, f"{label}_tail_std": math.nan}
    return {f"{label}_tail_mean": float(tail.mean()), f"{label}_tail_std": float(tail.std())}


def _checkpoint_info(artifact_root: Path, run_uuid: str, experiment_id: int) -> dict[str, object]:
    """Locate the best saved checkpoint on disk and return its name and size.

    Expects checkpoints under::

        {artifact_root}/{experiment_id}/{run_uuid}/artifacts/checkpoints/best/

    Takes the first ``.ckpt`` file alphabetically (relies on epoch-ordered
    naming convention). Returns None / NaN if the directory is empty or missing.

    :param artifact_root: Root of the MLflow artifact store.
    :param run_uuid: Full MLflow run UUID.
    :param experiment_id: Integer MLflow experiment ID.
    :return: Dict with keys ``"checkpoint"`` (filename str or None) and
             ``"checkpoint_mb"`` (file size in MB or NaN).
    """
    ckpt_dir = artifact_root / str(experiment_id) / run_uuid / "artifacts" / "checkpoints" / "best"
    checkpoints = sorted(ckpt_dir.glob("*.ckpt"))
    if not checkpoints:
        return {"checkpoint": None, "checkpoint_mb": math.nan}
    checkpoint = checkpoints[0]
    return {"checkpoint": checkpoint.name, "checkpoint_mb": checkpoint.stat().st_size / 1_000_000}


def _model_total_params(artifact_root: Path, run_uuid: str, experiment_id: int) -> str | None:
    """Extract the total parameter count string from the saved model summary.

    Reads ``artifacts/model/model_summary.txt`` for the run and finds the line
    containing ``"Total params"``, returning the value portion before that
    string (e.g. ``"12.3 M"`` from ``"12.3 M  Total params  ..."``).
    Returns None if the file does not exist or the line is not present.

    :param artifact_root: Root of the MLflow artifact store.
    :param run_uuid: Full MLflow run UUID.
    :param experiment_id: Integer MLflow experiment ID.
    :return: Parameter count string (e.g. ``"12.3 M"``) or None.
    """
    summary_path = artifact_root / str(experiment_id) / run_uuid / "artifacts" / "model" / "model_summary.txt"
    if not summary_path.exists():
        return None
    for line in summary_path.read_text().splitlines():
        if "Total params" in line:
            return line.split("Total params")[0].strip()
    return None


def best_per_group(df: pd.DataFrame, group_by: str = 'lr', rank_by: str = 'val_dice_tail_mean') -> "pd.io.formats.style.Styler":
    """Return a styled table of the best run per group, ranked by a tail-stability metric.

    For each unique value of ``group_by``, selects the single run with the
    highest ``rank_by`` value and formats the result as a Jupyter-ready Styler.
    Default grouping is by learning rate; pass ``group_by='encoder'`` or
    ``group_by='merge'`` to pivot the comparison.

    :param df: Output of ``summarize_runs`` — one row per run.
    :param group_by: Column to group by before picking the best run.
                     Must be a column in ``df``. Defaults to ``'lr'``.
    :param rank_by: Metric column used to select the winner within each group.
                    Defaults to ``'val_dice_tail_mean'`` (tail stability).
    :return: Styled DataFrame with one row per group value, sorted by
             ``rank_by`` descending.
    """
    cols = [
        group_by, 'encoder', 'merge',
        'val_dice_tail_mean',
        'val_dice_max',
        'val_dice_max_epoch',
        'generalization_gap_final',
        'samples_per_sec',
    ]
    result = (
        df
        .loc[df.groupby(group_by)[rank_by].idxmax(), cols]
        .sort_values(rank_by, ascending=False)
        .reset_index(drop=True)
    )
    return result.style.format({
        'val_dice_max': '{:.4f}',
        'val_dice_tail_mean': '{:.4f}',
        'generalization_gap_final': '{:.3f}',
        'samples_per_sec': '{:.1f}',
    })


def rank_all_runs(df: pd.DataFrame, group_by: str = 'lr', sort_by: str = 'val_dice_tail_mean') -> "pd.io.formats.style.Styler":
    """Return a styled full ranking table across all runs, with shortened column names.

    Selects the standard set of comparison columns, sorts by ``sort_by``
    descending, renames verbose MLflow column names to compact display aliases,
    and applies a ``RdYlGn`` gradient on the primary sort column. Designed for
    cross-LR (or cross-encoder / cross-merge) sweep summaries where ``df`` is
    the concatenated ``all_runs`` DataFrame with a ``group_by`` sweep column.

    :param df: Concatenated ``summarize_runs`` output with a ``group_by`` column
               added (e.g. ``'lr'``).
    :param group_by: Sweep-dimension column to include first (e.g. ``'lr'``,
                     ``'encoder'``). Defaults to ``'lr'``.
    :param sort_by: Column used for descending sort and gradient highlight.
                    Defaults to ``'val_dice_tail_mean'``.
    :return: Styled DataFrame with one row per run, sorted best-first.
    """
    rank_cols = [
        group_by, 'encoder', 'merge',
        'val_dice_max',
        'val_dice_tail_mean',
        'val_dice_tail_std',
        'val_dice_max_epoch',
        'generalization_gap_final',
        'drop_peak_to_final',
        'samples_per_sec',
    ]
    present = [c for c in rank_cols if c in df.columns]
    result = (
        df[present]
        .sort_values(sort_by, ascending=False)
        .reset_index(drop=True)
    )
    result = result.rename(columns={
        'val_dice_tail_std': 'tail_std',
        'val_dice_max_epoch': 'epoch',
        'generalization_gap_final': 'gen_gap',
        'drop_peak_to_final': 'drop',
    })
    fmt = {
        'val_dice_max': '{:.4f}',
        'val_dice_tail_mean': '{:.4f}',
        'tail_std': '{:.4f}',
        'gen_gap': '{:.3f}',
        'drop': '{:.3f}',
        'samples_per_sec': '{:.1f}',
    }
    gradient_col = sort_by if sort_by in result.columns else 'val_dice_tail_mean'
    return (
        result.style
        .format({k: v for k, v in fmt.items() if k in result.columns})
        .background_gradient(subset=[gradient_col], cmap='RdYlGn')
    )


def load_sweep_runs(
    experiments: dict[str, tuple[str, str]],
    mlruns: Path,
    group_by: str = 'lr',
    monitor: str = 'val_dice',
) -> pd.DataFrame:
    """Load and concatenate per-experiment MLflow databases into a single sweep DataFrame.

    Iterates over ``experiments``, loads each SQLite database with
    ``load_mlflow_tables``, builds a run summary via ``summarize_runs``, tags
    every row with the experiment's group label (e.g. its learning rate string),
    and returns the concatenated result. The group label becomes a plain string
    column named ``group_by`` so downstream functions (``best_per_group``,
    ``rank_all_runs``) can pivot by it without any further setup.

    :param experiments: Mapping of group label → ``(db_filename, experiment_name)``.
                        ``db_filename`` is resolved relative to ``mlruns``;
                        ``experiment_name`` is accepted but not used (reserved for
                        future filtering).
    :param mlruns: Root directory that contains the per-experiment ``.db`` files.
    :param group_by: Name of the column to add for the group label.
                     Defaults to ``'lr'``.
    :param monitor: Validation metric forwarded to ``summarize_runs``.
                    Defaults to ``'val_dice'``.
    :return: Concatenated DataFrame with one row per run and an added
             ``group_by`` column. Index is reset and contiguous.
    """
    frames = []
    for label, (db_file, _exp_name) in experiments.items():
        tables = load_mlflow_tables(mlruns / db_file)
        summary = summarize_runs(tables, monitor=monitor)
        summary[group_by] = label
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def pivot_dim_effect(
    df: pd.DataFrame,
    dim_col: str,
    group_col: str = 'lr',
    group_order: list[str] | None = None,
    dim_order: list[str] | None = None,
    value_col: str = 'val_dice_tail_mean',
    cmap: str = 'RdYlGn',
) -> "pd.io.formats.style.Styler":
    """Pivot mean and std of a metric across a sweep dimension and group axis.

    Produces a wide table where:

    - **rows** are ``group_col`` values (e.g. learning rates), ordered by
      ``group_order`` when provided.
    - **columns** are a two-level MultiIndex: outer level = ``dim_col`` values
      (e.g. encoder names), ordered by ``dim_order``; inner level = ``mean``
      and ``std`` side by side.

    Background gradient is applied to the ``mean`` columns only (``axis=None``
    so colours are normalised across the whole mean sub-table, not per row).

    Example — ``pivot_dim_effect(all_runs, 'encoder', group_order=LR_ORDER)``
    produces::

        encoder       classical          se             he2
                    mean    std    mean    std    mean    std
        lr
        1e-4       0.8361  0.0014  0.8289  ...   0.8312  ...
        3e-4       0.8502  0.0011  0.8422  ...   0.8461  ...
        ...

    :param df: Sweep DataFrame produced by ``load_sweep_runs``.
    :param dim_col: Column whose values form the outer column level
                    (e.g. ``'encoder'`` or ``'merge'``).
    :param group_col: Column whose values form the row index (e.g. ``'lr'``).
    :param group_order: Explicit row ordering. Rows absent from ``df`` become
                        NaN rows. If ``None``, sorted alphabetically.
    :param dim_order: Explicit column ordering for ``dim_col`` values. Columns
                      absent from ``df`` become NaN columns. If ``None``,
                      sorted alphabetically.
    :param value_col: Metric to aggregate. Defaults to ``'val_dice_tail_mean'``.
    :param cmap: Colormap for the mean gradient. Defaults to ``'RdYlGn'``.
    :return: Styled DataFrame ready for Jupyter display.
    """
    mean_piv = df.groupby([group_col, dim_col])[value_col].mean().unstack(dim_col)
    std_piv = df.groupby([group_col, dim_col])[value_col].std().unstack(dim_col)

    if group_order:
        mean_piv = mean_piv.reindex(group_order)
        std_piv = std_piv.reindex(group_order)
    cols = dim_order or sorted(mean_piv.columns.tolist())
    mean_piv = mean_piv.reindex(columns=cols)
    std_piv = std_piv.reindex(columns=cols)

    # Interleave: (col, 'mean'), (col, 'std') for each dim value
    combined = pd.concat(
        {c: pd.DataFrame({'mean': mean_piv[c], 'std': std_piv[c]}) for c in cols},
        axis=1,
    )
    combined.index.name = group_col

    mean_subset = [(c, 'mean') for c in cols]
    return (
        combined.style
        .format('{:.4f}')
        .background_gradient(subset=mean_subset, axis=None, cmap=cmap)
    )


def arch_consistency(
    df: pd.DataFrame,
    group_cols: list[str] | tuple[str, ...] = ('encoder', 'merge'),
    value_col: str = 'val_dice_tail_mean',
    cmap: str = 'RdYlGn',
) -> "pd.io.formats.style.Styler":
    """Summarise cross-LR consistency of each architecture by mean, std, min, max.

    Groups ``df`` by ``group_cols`` (default: encoder × merge) and aggregates
    ``value_col`` with mean, std, min, and max. Rows are sorted by mean
    descending so the most consistent high-performing architecture appears first.
    Background gradient is applied to the mean column only.

    Low ``std`` combined with high ``mean`` identifies architectures that are
    both accurate and robust to learning-rate choice — the key signal this table
    is designed to surface.

    :param df: Sweep DataFrame produced by ``load_sweep_runs``.
    :param group_cols: Columns that identify an architecture. Defaults to
                       ``('encoder', 'merge')``.
    :param value_col: Metric to aggregate. Defaults to ``'val_dice_tail_mean'``.
    :param cmap: Colormap for the mean gradient. Defaults to ``'RdYlGn'``.
    :return: Styled DataFrame with columns ``[*group_cols, mean, std, min, max]``,
             sorted by mean descending.
    """
    group_cols = list(group_cols)
    result = (
        df.groupby(group_cols)[value_col]
        .agg(['mean', 'std', 'min', 'max'])
        .sort_values('mean', ascending=False)
        .reset_index()
    )
    metric = value_col.replace('val_dice_', '').replace('val_', '')
    result.columns = group_cols + [
        f'mean_{metric}', f'std_{metric}', f'min_{metric}', f'max_{metric}',
    ]
    fmt_cols = [c for c in result.columns if c not in group_cols]
    return (
        result.style
        .format({c: '{:.4f}' for c in fmt_cols})
        .background_gradient(subset=[f'mean_{metric}'], cmap=cmap)
    )
