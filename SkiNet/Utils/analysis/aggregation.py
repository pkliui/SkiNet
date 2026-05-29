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
      ``final_val_best_dice_at_threshold``, ``final_val_iou``, ``final_val_loss``,
      ``samples_per_sec``, ``time_per_step_ms``, ``final_grad_scale``
    - Best-epoch per metric: ``best_{monitor}``, ``best_val_dice``, ``best_val_iou``,
      ``min_val_loss`` — each with ``_epoch`` and ``_step`` siblings
    - Tail stability per metric: ``{label}_tail_mean``, ``{label}_tail_std``
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
        # - best_val_best_dice_at_threshold      → peak dice across all epochs (ceiling)
        # - best_val_best_dice_at_threshold_epoch→ which epoch that peak occurred at
        # - best_val_dice_tail_mean              → mean dice over last 10 epochs (stability)
        # - best_val_dice_tail_std               → std over last 10 epochs (noise level)
        # - generalization_gap_final             → train_dice - val_dice at final epoch (overfitting signal)

    :param tables: Output of ``load_mlflow_tables`` — dict with keys
                   ``"runs"``, ``"metrics"``, ``"latest"``.
    :param artifact_root: Optional MLflow artifact directory root. When given,
                          checkpoint info and model param count are added.
    :param monitor: Validation metric used for checkpoint selection and primary
                    sort key. Defaults to ``"val_best_dice_at_threshold"``.
    :param tail_n: Number of final epochs over which to compute tail mean and
                   std stability columns. Defaults to 10.
    :return: DataFrame with one row per run, sorted by ``best_{monitor}``
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
            "final_val_best_dice_at_threshold": _latest_value(run_latest, "final/val_best_dice_at_threshold"),
            "final_val_iou": _latest_value(run_latest, "final/val_iou"),
            "final_val_loss": _latest_value(run_latest, "final/val_loss"),
            "samples_per_sec": _latest_value(run_latest, "final/perf/samples_per_sec"),
            "time_per_step_ms": _latest_value(run_latest, "final/perf/time_per_step_ms"),
            "final_grad_scale": _latest_value(run_latest, "final/grad_scale"),
        }

        # find the best value for each of the metrics of interest, along with the epoch and step at which it occurred
        for key, label, mode in (
            (monitor, f"best_{monitor}", "max"),
            ("val_dice", "best_val_dice", "max"),
            ("val_iou", "best_val_iou", "max"),
            ("val_loss", "min_val_loss", "min"),
        ):
            record.update(_best_metric_columns(run_metrics, key, label, mode))
            record.update(_tail_metric_columns(run_metrics, key, label, tail_n))

        record["generalization_gap_final"] = record["final_train_dice"] - record["final_val_dice"]
        # how far the run fell from its best-epoch score to its last-epoch score
        record["drop_peak_to_final"] = record[f"best_{monitor}"] - record.get(f"final_{monitor}", float("nan"))
        # how far the single best epoch sits above the tail-mean plateau
        record["gap_peak_to_tail"] = record[f"best_{monitor}"] - record.get(f"best_{monitor}_tail_mean", float("nan"))

        if artifact_root is not None:
            record.update(_checkpoint_info(artifact_root, run_uuid, int(run["experiment_id"])))
            record["model_total_params"] = _model_total_params(artifact_root, run_uuid, int(run["experiment_id"]))
        records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.sort_values(f"best_{monitor}", ascending=False).reset_index(drop=True)


def rank_table(
    run_summary: pd.DataFrame,
    monitor: str = "val_best_dice_at_threshold",
    sort_by: str = "best",
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

    sort_col = f"best_{monitor}" if sort_by == "best" else f"best_{monitor}_tail_mean"
    sort_label = (
        "peak value across all epochs"
        if sort_by == "best"
        else f"tail mean over last {tail_n} epochs"
    )
    cols = [
        "encoder", "merge", "run_short", "status",
        f"best_{monitor}", f"best_{monitor}_tail_mean", f"best_{monitor}_tail_std",
        f"best_{monitor}_epoch",
        "best_val_dice", "best_val_iou", "min_val_loss",
        f"final_{monitor}", "final_val_dice",
        "drop_peak_to_final", "gap_peak_to_tail",
        "generalization_gap_final", "samples_per_sec", "duration_min",
        "checkpoint", "checkpoint_mb", "model_total_params",
    ]
    present = [c for c in cols if c in run_summary.columns]
    fmt = {
        f"best_{monitor}": "{:.4f}",
        f"best_{monitor}_tail_mean": "{:.4f}",
        f"best_{monitor}_tail_std": "{:.4f}",
        "best_val_dice": "{:.4f}",
        "best_val_iou": "{:.4f}",
        "min_val_loss": "{:.4f}",
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
    df = run_summary[present].sort_values(sort_col, ascending=False) if sort_col in present else run_summary[present]

    def _highlight_top3(s: pd.Series) -> list[str]:
        return ["background-color: #e8f5e9" if i < 3 else "" for i in range(len(s))]

    return (
        df.style.format({k: v for k, v in fmt.items() if k in present})
        .apply(_highlight_top3, axis=0)
        .set_caption(caption)
    )


def family_summary(run_summary: pd.DataFrame, group_cols: Iterable[str] = ("encoder", "merge")) -> pd.DataFrame:
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
        grouped = (
            run_summary.groupby(group_col)
            .agg(
                n=("run_uuid", "count"),
                mean_best_dice=("best_val_best_dice_at_threshold", "mean"),
                max_best_dice=("best_val_best_dice_at_threshold", "max"),
                mean_final_dice=("final_val_best_dice_at_threshold", "mean"),
                mean_samples_per_sec=("samples_per_sec", "mean"),
                mean_gap=("generalization_gap_final", "mean"),
            )
            .assign(family=group_col)
            .reset_index(names="value")
        )
        summaries.append(grouped)
    df = pd.concat(summaries, ignore_index=True).sort_values(["family", "mean_best_dice"], ascending=[True, False])
    return df.style.format({
        'mean_best_dice': '{:.4f}',
        'max_best_dice': '{:.4f}',
        'mean_final_dice': '{:.4f}',
        'mean_samples_per_sec': '{:.1f}',
        'mean_gap': '{:.4f}',
    })


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
