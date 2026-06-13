"""Learning-rate sweep helpers — load and compare runs across MLflow DBs.

These support the E1 model/LR sweep: comparing encoder × merge architectures
across a learning-rate sweep, where the LR is the grouping axis and each LR has
its own MLflow database.

``load_sweep_runs`` loads one MLflow database per sweep point (one DB per
learning rate), summarises each with
:func:`~SkiNet.Utils.analysis.aggregation.summarize_runs`, tags every row with
its sweep-dimension label (``lr`` by default), and concatenates them into one
tall DataFrame (one row per run). The remaining functions are the comparison
tables built on top of that DataFrame.

Typical pipeline (cross-LR model sweep, see the
``E1-...-modelsw-summary-all-lr`` notebook)::

    all_runs = load_sweep_runs(EXPERIMENTS, MLRUNS, monitor="val_dice")
    best_run_per_group(all_runs, group_by="lr")      # winner per LR
    rank_all_runs(all_runs, group_by="lr")           # every run, ranked
    pivot_dim_effect(all_runs, dim_col="encoder")    # LR × encoder grid
    arch_consistency(all_runs)                        # robustness to LR
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from SkiNet.Utils.analysis.aggregation import summarize_runs
from SkiNet.Utils.analysis.io import load_mlflow_tables


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_sweep_runs(
    experiments: dict[str, tuple[str, str]],
    mlruns: Path,
    monitor: str,
    group_by: str = 'lr',
) -> pd.DataFrame:
    """Load and concatenate per-experiment MLflow databases into a single sweep DataFrame.

    Iterates over ``experiments``, loads each SQLite database with
    ``load_mlflow_tables``, builds a run summary via ``summarize_runs``, tags
    every row with the experiment's group label (e.g. its learning rate string),
    and returns the concatenated result. The group label becomes a plain string
    column named ``group_by`` so downstream functions (``best_run_per_group``,
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


# ---------------------------------------------------------------------------
# Ranking tables
# ---------------------------------------------------------------------------

def best_run_per_group(df: pd.DataFrame, group_by: str = 'lr', rank_by: str = 'val_dice_tail_mean') -> "pd.io.formats.style.Styler":
    """Return a styled table of the single best run per group, ranked by a tail-stability metric.

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


# ---------------------------------------------------------------------------
# Cross-dimension pivots
# ---------------------------------------------------------------------------

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
