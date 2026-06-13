"""Learning-rate decay study helpers — E3 constant vs ReduceLROnPlateau vs cosine.

These back the ``E3-...-lr-decay-study`` notebook, which asks a single question:
**should we use LR decay at all** on the locked ``classical+attention_gate``
architecture at ``lr=3e-4``, or is constant LR (the E2 baseline) sufficient?

Unlike :mod:`SkiNet.Utils.analysis.lr_sweep` (which compares architectures across
a *learning-rate sweep*, one DB per LR), this module compares **decay schedules**
on a single seed: constant LR (E2), ReduceLROnPlateau, and CosineAnnealingLR, each
in its own MLflow database. The pipeline mirrors the rest of the analysis package —
notebook cells call these functions and contain no inline logic::

    conds   = load_decay_conditions(CONDITIONS, monitor=MONITOR, seed=SEED)
    show_decay_config(conds, 'cosine')              # § config audit
    show_decay_comparison(conds, baseline='constant')   # § 3-way table
    show_decay_milestones(conds, 'cosine', lr_max=3e-4) # § LR milestones
    plot_decay_dynamics(conds, ...)                  # § figure

Each *condition* is a :class:`DecayCondition`: the per-run summary row plus the
raw per-epoch series (monitor, learning rate, train Dice) needed for the figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.figure import Figure

from SkiNet.Utils.analysis.aggregation import summarize_runs
from SkiNet.Utils.analysis.io import load_mlflow_tables

LR_KEY = "lr-Adam"
TRAIN_DICE_KEY = "train_dice"


# ---------------------------------------------------------------------------
# Condition container
# ---------------------------------------------------------------------------

@dataclass
class DecayCondition:
    """One LR-schedule condition: its summary row plus the per-epoch series for figures.

    :ivar key: Short machine label (e.g. ``'constant'``, ``'rop'``, ``'cosine'``).
    :ivar label: Human display label used in tables and legends.
    :ivar colour: Hex colour used consistently across all figures.
    :ivar summary: The single ``summarize_runs`` row for this condition's run.
    :ivar monitor: Per-epoch validation monitor series (columns ``epoch``, ``value``).
    :ivar lr: Per-epoch learning-rate series (columns ``epoch``, ``value``).
    :ivar train_dice: Per-epoch train Dice series (columns ``epoch``, ``value``).
    :ivar params: Run parameters indexed by key (for the config audit).
    """

    key: str
    label: str
    colour: str
    summary: pd.Series
    monitor: pd.DataFrame
    lr: pd.DataFrame
    train_dice: pd.DataFrame
    params: pd.Series


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _epoch_series(metrics: pd.DataFrame, run_uuid: str, key: str) -> pd.DataFrame:
    """Return a metric's per-epoch series for one run as a tidy ``epoch``/``value`` frame.

    Rows are ordered by logging step and numbered ``epoch = 1..n`` to match the
    epoch convention used by ``summarize_runs`` (which counts logged points).

    :param metrics: Raw MLflow metrics table (all runs).
    :param run_uuid: Run to filter to.
    :param key: Metric key to extract (e.g. ``'lr-Adam'``).
    :return: DataFrame with columns ``epoch`` (1-based) and ``value``.
    """
    series = (
        metrics[(metrics["run_uuid"] == run_uuid) & (metrics["key"] == key)]
        .sort_values("step")
        .reset_index(drop=True)
    )
    return pd.DataFrame({"epoch": np.arange(1, len(series) + 1), "value": series["value"].to_numpy()})


def load_decay_conditions(
    conditions: dict[str, dict],
    monitor: str,
    seed: int,
    tail_n: int = 10,
) -> dict[str, DecayCondition]:
    """Load every LR-schedule condition into a dict of :class:`DecayCondition`.

    For each entry the database is loaded once, ``summarize_runs`` builds the
    per-run summary, and the seed-100 run is selected. Multi-run databases (e.g.
    the E2 5-seed baseline) are filtered by a ``seed`` regex on the run name and
    an optional ``merge`` architecture tag; single-run diagnostic databases take
    the only row. The per-epoch monitor / LR / train-Dice series are attached for
    the figure functions.

    :param conditions: Ordered mapping ``key -> spec`` where ``spec`` carries:
        ``label`` (display name), ``colour`` (hex), ``db`` (:class:`~pathlib.Path`
        to the MLflow SQLite file), and optionally ``merge`` (architecture tag to
        select a row from a multi-run baseline DB; omit for single-run DBs).
    :param monitor: Validation metric forwarded to ``summarize_runs`` and used as
        the per-epoch monitor series.
    :param seed: Seed to select from multi-run databases.
    :param tail_n: Tail-window length forwarded to ``summarize_runs``.
    :return: Mapping ``key -> DecayCondition`` preserving ``conditions`` order.
    """
    out: dict[str, DecayCondition] = {}
    for key, spec in conditions.items():
        tables = load_mlflow_tables(spec["db"])
        summary = summarize_runs(tables, monitor=monitor, tail_n=tail_n)

        if len(summary) == 1 and "merge" not in spec:
            row = summary.iloc[0]
        else:
            run_names = tables["runs"][["run_uuid", "run_name"]].copy()
            run_names["seed"] = run_names["run_name"].str.extract(r"seed(\d+)").astype(int)
            summary = summary.merge(run_names[["run_uuid", "seed"]], on="run_uuid")
            mask = summary["seed"] == seed
            if "merge" in spec:
                mask &= summary["merge"] == spec["merge"]
            row = summary[mask].iloc[0]

        uuid = row["run_uuid"]
        metrics = tables["metrics"]
        params = (
            tables["params"][tables["params"]["run_uuid"] == uuid]
            .set_index("key")["value"]
        )
        out[key] = DecayCondition(
            key=key,
            label=spec["label"],
            colour=spec["colour"],
            summary=row,
            monitor=_epoch_series(metrics, uuid, monitor),
            lr=_epoch_series(metrics, uuid, LR_KEY),
            train_dice=_epoch_series(metrics, uuid, TRAIN_DICE_KEY),
            params=params,
        )
    return out


# ---------------------------------------------------------------------------
# 3-way comparison table
# ---------------------------------------------------------------------------

_COMPARISON_COLS = [
    "plateau_dice", "plateau_std", "gen_gap", "sps", "dur_min",
]


def build_decay_comparison(
    conditions: dict[str, DecayCondition],
    monitor: str,
    baseline: str,
) -> pd.DataFrame:
    """Assemble the 3-way plateau-Dice comparison table with deltas vs the baseline.

    Plateau (tail-window mean) Dice is the pre-registered E3 decision metric;
    peak Dice is deliberately excluded — at a single seed it is dominated by
    seed-to-seed noise and is not the basis for any E3 decision. One row per
    condition (plateau mean/std, final generalisation gap, throughput, duration)
    and signed deltas of plateau Dice and throughput against ``baseline``.

    :param conditions: Output of :func:`load_decay_conditions`.
    :param monitor: Validation metric whose ``_tail_*`` columns are read.
    :param baseline: Condition key to compute deltas against (e.g. ``'constant'``).
    :return: Plain (unstyled) DataFrame indexed by condition label.
    """
    m = monitor
    src_cols = [
        f"{m}_tail_mean", f"{m}_tail_std",
        "generalization_gap_final", "samples_per_sec", "duration_min",
    ]
    table = pd.DataFrame(
        {c.label: c.summary[src_cols].to_numpy() for c in conditions.values()},
        index=_COMPARISON_COLS,
    ).T
    table.index.name = "condition"

    base_label = conditions[baseline].label
    table["Δplateau_vs_base"] = table["plateau_dice"] - table.loc[base_label, "plateau_dice"]
    table["Δsps_vs_base"] = table["sps"] - table.loc[base_label, "sps"]
    return table


def show_decay_comparison(
    conditions: dict[str, DecayCondition],
    monitor: str,
    baseline: str,
) -> None:
    """Display the 3-way comparison as a gradient-styled table.

    Thin presentation wrapper over :func:`build_decay_comparison`: formats each
    column and shades ``plateau_dice`` with ``RdYlGn``.

    :param conditions: Output of :func:`load_decay_conditions`.
    :param monitor: Validation metric forwarded to :func:`build_decay_comparison`.
    :param baseline: Condition key to compute deltas against.
    """
    table = build_decay_comparison(conditions, monitor, baseline)
    display(
        table.style.format({
            "plateau_dice": "{:.4f}", "plateau_std": "{:.4f}",
            "gen_gap": "{:.4f}", "sps": "{:.1f}", "dur_min": "{:.1f}",
            "Δplateau_vs_base": "{:+.4f}", "Δsps_vs_base": "{:+.1f}",
        }).background_gradient(subset=["plateau_dice"], cmap="RdYlGn")
    )

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_decay_dynamics(
    conditions: dict[str, DecayCondition],
    monitor: str,
    decay_key: str,
    title: str,
    save_path: Path | None = None,
    tail_n: int = 10,
) -> Figure:
    """Plot training curves (left) and LR schedules (log scale, right) for all conditions.

    Left panel overlays each condition's monitor curve and annotates its
    **plateau** — the tail-``tail_n`` mean (the pre-registered E3 decision
    metric) — as a shaded band over the final window. Peak markers are
    deliberately omitted: at a single seed the peak is noise-dominated and is not
    an E3 decision quantity. Right panel steps each condition's LR schedule on a
    log axis and highlights each decaying schedule's LR at **plateau onset** (the
    start of the tail window), the LR regime that produced the plateau Dice.

    :param conditions: Output of :func:`load_decay_conditions` (drawn in order).
    :param monitor: Validation metric, used for the left-panel y-label.
    :param decay_key: Condition whose plateau-onset LR is labelled on the right panel.
    :param title: Figure suptitle.
    :param save_path: If given, the figure is written here at 130 dpi.
    :param tail_n: Length of the plateau (tail) window; must match ``load_decay_conditions``.
    :return: The created :class:`~matplotlib.figure.Figure`.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for cond in conditions.values():
        ax1.plot(cond.monitor["epoch"], cond.monitor["value"],
                 color=cond.colour, lw=1.8, alpha=0.85, label=cond.label)
        n_ep = int(len(cond.monitor))
        plateau_start = n_ep - tail_n + 1
        plateau_mean = cond.summary[f"{monitor}_tail_mean"]
        ax1.hlines(plateau_mean, plateau_start, n_ep,
                   color=cond.colour, lw=2.6, zorder=5)
        ax1.axvspan(plateau_start, n_ep, color=cond.colour, alpha=0.06, zorder=0)
        ax1.annotate(f"plateau {plateau_mean:.4f}", (plateau_start, plateau_mean),
                     xytext=(plateau_start - 70, plateau_mean), fontsize=8, color=cond.colour,
                     va="center", arrowprops=dict(arrowstyle="->", color=cond.colour, lw=0.8))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(monitor)
    ax1.set_title(f"a — validation curves with tail-{tail_n} plateau (seed 100)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(ls=":", alpha=0.4)
    ax1.set_xlim(0, 200)

    for cond in conditions.values():
        ax2.step(cond.lr["epoch"], cond.lr["value"], where="post",
                 color=cond.colour, lw=2.0, alpha=0.9, label=cond.label)
    decay = conditions[decay_key]
    plateau_start = int(len(decay.monitor)) - tail_n + 1
    lr_at_plateau = decay.lr.iloc[plateau_start - 1]["value"]
    ax2.scatter([plateau_start], [lr_at_plateau], color=decay.colour, s=70, zorder=5,
                label=f"{decay.label} plateau onset (ep {plateau_start}, LR={lr_at_plateau:.2e})")
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0e}" if x < 1e-4 else f"{x:.1e}"))
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning rate (log scale)")
    ax2.set_title("b — LR schedules", fontsize=11)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(ls=":", alpha=0.4)
    ax2.set_xlim(0, 200)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig
