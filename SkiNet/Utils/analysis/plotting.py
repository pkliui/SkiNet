"""Matplotlib/seaborn plotting helpers for MLflow experiment visualisation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from SkiNet.Utils.analysis.schema import ARCH, SEED


# ── Global style ───────────────────────────────────────────────────────────────

def set_paper_style(context: str = "notebook", font_scale: float = 1.0) -> None:
    """Apply a Tufte-inspired minimal theme for report-quality figures.

    Follows the data-ink maximisation principle from *Beautiful Evidence*:
    no grid, no top/right spines, muted axis lines, serif body text.
    Affects global matplotlib ``rcParams`` for the current process only.

    :param context: Seaborn context controlling element scaling
                    (``"paper"``, ``"notebook"``, ``"talk"``, ``"poster"``).
    :param font_scale: Multiplier applied to all font sizes.
    """
    sns.set_theme(
        style="ticks",
        context=context,
        font_scale=font_scale,
        rc={
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            # axes — only bottom + left; muted so they recede behind data
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#888888",
            "axes.linewidth": 0.6,
            # no grid — data speaks without scaffolding
            "axes.grid": False,
            # typography
            "axes.titleweight": "regular",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.color": "#555555",
            "ytick.color": "#555555",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        },
    )


# ── Shared colour helper ──────────────────────────────────────────────────────

def _tail_mean_palette(
    run_summary: pd.DataFrame,
    monitor: str,
) -> dict[str, str]:
    """Return a {architecture: hex_colour} map using RdYlGn ranked by tail-mean.

    Best tail-mean → darkest green (#1b5e20 end of RdYlGn).
    Worst           → red end.
    Uses [0.10, 0.90] of the colormap to keep colours vivid and distinguishable.
    """
    tail_col = f"{monitor}_tail_mean"
    rs = run_summary.copy()
    rs["_arch"] = rs["encoder"] + " + " + rs["merge"]
    if tail_col not in rs.columns:
        # Fallback: evenly spaced colours
        archs = rs["_arch"].tolist()
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        return {a: palette[i % len(palette)] for i, a in enumerate(archs)}
    vmin, vmax = rs[tail_col].min(), rs[tail_col].max()
    span = vmax - vmin if vmax > vmin else 1.0
    return {
        row["_arch"]: mcolors.to_hex(cm.get_cmap("RdYlGn")(0.10 + 0.80 * (row[tail_col] - vmin) / span))
        for _, row in rs.iterrows()
    }


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_architecture_heatmap(run_summary: pd.DataFrame, value_col: str, title: str) -> Figure:
    """Plot encoder x merge heatmap for a run-level metric.

    :param run_summary: DataFrame with columns ``encoder``, ``merge``, and ``value_col``.
                        Each ``(encoder, merge)`` pair must be unique; duplicates raise
                        ``ValueError`` from ``pivot``.
    :param value_col: Column name to use as cell values (e.g. ``"best_val_dice"``).
    :param title: Axes title.
    :return: Matplotlib Figure.
    """
    heat = run_summary.pivot(index="encoder", columns="merge", values=value_col)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(heat.values, cmap="RdYlGn")
    ax.set_xticks(range(len(heat.columns)), heat.columns)
    ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_xlabel("Merge residual mode")
    ax.set_ylabel("Encoder residual mode")
    ax.set_title(title)
    vmin_h, vmax_h = np.nanmin(heat.values), np.nanmax(heat.values)
    for row_idx in range(heat.shape[0]):
        for col_idx in range(heat.shape[1]):
            val = heat.iloc[row_idx, col_idx]
            label = f"{val:.4f}" if pd.notna(val) else "n/a"
            # Dark text on light cells, light text on dark cells
            norm = (val - vmin_h) / (vmax_h - vmin_h + 1e-10) if pd.notna(val) else 0.5
            rgba = cm.get_cmap("RdYlGn")(0.10 + 0.80 * norm)
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "#000000" if lum > 0.45 else "#ffffff"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color=text_color, fontweight="bold")
    fig.colorbar(image, ax=ax, label=value_col)
    fig.tight_layout()
    return fig


def plot_learning_curves(
    history: pd.DataFrame,
    key: str,
    ylabel: str,
    title: str,
    monitor: str | None = None,
    run_summary: pd.DataFrame | None = None,
) -> Figure:
    """Plot one metric history curve per architecture, coloured by tail-mean rank.

    :param history: Long-format DataFrame with columns ``architecture``, ``key``,
                    ``epoch``, ``value``.
    :param key: Metric name to filter on (e.g. ``"val_dice"``).
    :param ylabel: Y-axis label.
    :param title: Axes title.
    :param monitor: Base metric name used to derive the tail-mean column for the
                    colour ranking. Required only when ``run_summary`` is given.
    :param run_summary: If provided (together with ``monitor``), line colours
                        follow the RdYlGn scale ranked by tail-mean Dice
                        (best = green, worst = red).
    :return: Matplotlib Figure.
    """
    if run_summary is not None and monitor is None:
        raise ValueError("monitor is required when run_summary is provided")
    colour_map = _tail_mean_palette(run_summary, monitor) if run_summary is not None and monitor is not None else {}

    fig, ax = plt.subplots(figsize=(10, 5.2))
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (architecture, sub) in enumerate(history[history["key"].eq(key)].groupby("architecture")):
        color = colour_map.get(architecture, palette[i % len(palette)])
        ax.plot(sub["epoch"], sub["value"], label=architecture, linewidth=1.8, color=color)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_train_val_overlay(
    history: pd.DataFrame,
    monitor: str,
    title: str = "Train vs validation Dice by epoch",
    run_summary: pd.DataFrame | None = None,
) -> Figure:
    """Plot train Dice (dashed) and val Dice (solid) together per architecture.

    Colour encodes architecture via RdYlGn ranked by tail-mean when ``run_summary``
    is provided; line style encodes train vs validation.

    :param history: Output of ``epoch_metrics`` (long-format).
    :param title: Axes title.
    :param run_summary: If provided, colours follow RdYlGn by tail-mean rank.
    :param monitor: Base metric name for tail-mean lookup.
    :return: Matplotlib Figure.
    """
    colour_map = _tail_mean_palette(run_summary, monitor) if run_summary is not None else {}

    fig, ax = plt.subplots(figsize=(10, 5.2))
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    architectures = sorted(history["architecture"].unique())

    for idx, architecture in enumerate(architectures):
        color = colour_map.get(architecture, palette[idx % len(palette)])
        val_sub = history[(history["architecture"] == architecture) & (history["key"] == "val_dice")]
        train_sub = history[(history["architecture"] == architecture) & (history["key"] == "train_dice")]
        if not val_sub.empty:
            ax.plot(val_sub["epoch"], val_sub["value"], color=color, linewidth=1.8, label=architecture)
        if not train_sub.empty:
            ax.plot(train_sub["epoch"], train_sub["value"], color=color, linewidth=1.0, linestyle="--")

    val_handle = mlines.Line2D([], [], color="grey", linewidth=1.8, label="Validation")
    train_handle = mlines.Line2D([], [], color="grey", linewidth=1.0, linestyle="--", label="Train")

    arch_handles, arch_labels = ax.get_legend_handles_labels()
    arch_legend = ax.legend(arch_handles, arch_labels, fontsize=8,
                            bbox_to_anchor=(1.02, 1), loc="upper left", title="Architecture")
    ax.add_artist(arch_legend)
    ax.legend(handles=[val_handle, train_handle], loc="lower right")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    fig.tight_layout()
    return fig


def plot_group_bar(
    df: pd.DataFrame,
    group_col: str = 'lr',
    group_order: list[str] | None = None,
    arch_cols: tuple[str, str] = ('encoder', 'merge'),
    value_col: str = 'val_dice_tail_mean',
    title: str | None = None,
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (14, 5),
    bar_width: float = 0.18,
    cmap: str = 'Blues',
) -> Figure:
    """Bar chart of a metric per architecture, with one bar cluster per group value.

    Architectures are ordered on the x-axis by their mean ``value_col`` across
    all groups (best architecture left-most). Each group gets a distinct colour
    sampled from ``cmap``, using only the upper portion of the scale so colours
    stay distinguishable against a white background.

    :param df: Sweep DataFrame with ``group_col``, ``arch_cols[0]``,
               ``arch_cols[1]``, and ``value_col`` columns.
    :param group_col: Column whose distinct values define the bar clusters
                      (e.g. ``'lr'``).
    :param group_order: Explicit ordering for the group values. If ``None``,
                        groups are sorted alphabetically.
    :param arch_cols: Two-column tuple used to build the architecture label
                      (encoder col, merge col). Defaults to
                      ``('encoder', 'merge')``.
    :param value_col: Metric to plot. Defaults to ``'val_dice_tail_mean'``.
    :param title: Axes title. Defaults to a generated string.
    :param ylabel: Y-axis label. Defaults to ``value_col``.
    :param ylim: Optional ``(ymin, ymax)`` tuple. If ``None``, matplotlib
                 auto-scales.
    :param figsize: Figure size in inches.
    :param bar_width: Width of each individual bar.
    :param cmap: Matplotlib colormap name for group colours. A sequential map
                 (e.g. ``'Blues'``, ``'viridis'``) keeps things calm; avoid
                 qualitative maps with saturated primaries.
    :return: Matplotlib Figure.
    """
    enc_col, mrg_col = arch_cols
    groups = group_order or sorted(df[group_col].unique())
    n_groups = len(groups)

    # colour: sample the upper [0.35, 0.90] range so all bars are visible
    colour_map = cm.get_cmap(cmap)
    colours = [colour_map(0.35 + 0.55 * i / max(n_groups - 1, 1)) for i in range(n_groups)]

    arch_order = (
        df.groupby([enc_col, mrg_col])[value_col]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    arch_labels = [f'{e}\n{m}' for e, m in arch_order]
    n_arch = len(arch_order)
    x = np.arange(n_arch)

    fig, ax = plt.subplots(figsize=figsize)
    for i, group in enumerate(groups):
        subset = df[df[group_col] == group].set_index([enc_col, mrg_col])
        vals = [
            float(subset.loc[combo, value_col]) if combo in subset.index else np.nan
            for combo in arch_order
        ]
        ax.bar(x + i * bar_width, vals, bar_width,
               label=f'{group_col}={group}', color=colours[i], alpha=0.9)

    ax.set_xticks(x + bar_width * (n_groups - 1) / 2)
    ax.set_xticklabels(arch_labels, fontsize=8)
    ax.set_ylabel(ylabel or value_col, fontsize=10)
    ax.set_title(title or f'{value_col} by architecture, grouped by {group_col}')
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    ax.legend(title=group_col, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


def plot_accuracy_throughput(
    run_summary: pd.DataFrame,
    monitor: str,
) -> Figure:
    """Plot accuracy-throughput frontier coloured by tail-mean rank (RdYlGn).

    Peak values are shown as hollow markers; tail-mean ± std as filled markers.
    Both marker sets share the same RdYlGn colour keyed to tail-mean rank so the
    colour meaning is consistent with learning-curve plots.

    :param run_summary: DataFrame with columns ``encoder``, ``merge``,
                        ``samples_per_sec``, and the derived columns above.
    :param monitor: Base metric name used to construct column lookups.
    :return: Matplotlib Figure.
    """
    best_col = f"{monitor}_max"
    tail_col = f"{monitor}_tail_mean"
    std_col = f"{monitor}_tail_std"
    has_tail = tail_col in run_summary.columns
    has_std = std_col in run_summary.columns

    colour_map = _tail_mean_palette(run_summary, monitor)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    color_handles, color_labels = [], []

    for _, row in run_summary.iterrows():
        arch = f"{row['encoder']} + {row['merge']}"
        color = colour_map.get(arch, "#888888")

        # Peak — hollow marker
        ax.scatter(row["samples_per_sec"], row[best_col], s=100,
                   facecolors="none", edgecolors=color, linewidths=1.8, zorder=3)

        ax.annotate(arch, (float(row["samples_per_sec"]), float(row[best_col])),
                    xytext=(5, 4), textcoords="offset points", fontsize=7, color=color)

        if has_tail:
            # Tail mean — filled marker
            ax.scatter(row["samples_per_sec"], row[tail_col], s=100, color=color, zorder=4)
            if has_std:
                ax.errorbar(row["samples_per_sec"], row[tail_col], yerr=row[std_col],
                            fmt="none", ecolor=color, capsize=4, linewidth=1.2, zorder=2)

        color_handles.append(
            mlines.Line2D([], [], marker="o", linestyle="none", markersize=8, color=color, label=arch)
        )
        color_labels.append(arch)

    peak_handle = mlines.Line2D([], [], marker="o", linestyle="none", markersize=8,
                                markerfacecolor="none", markeredgecolor="grey", label="Peak")
    tail_handle = mlines.Line2D([], [], marker="o", linestyle="none", markersize=8,
                                color="grey", label="Tail mean ± std")

    first_legend = ax.legend(color_handles, color_labels, title="Architecture (colour = tail-mean rank)",
                             loc="lower right", fontsize=7)
    ax.add_artist(first_legend)
    ax.legend(handles=[peak_handle, tail_handle], loc="upper left")

    ax.set_title("Accuracy-throughput frontier  [colour = RdYlGn by tail-mean rank]")
    ax.set_xlabel("Final training throughput, samples/sec")
    ax.set_ylabel("Validation Dice-at-threshold")
    fig.tight_layout()
    return fig


# ---------- E2 model sweep notebooks ------------


def plot_paired_slopegraph(
    runs: pd.DataFrame,
    metrics: list[tuple[str, str]],
    *,
    arch_a: str,
    arch_b: str,
    seeds: list[int],
    palette: dict[str, str],
    seed_col: str = SEED,
    arch_col: str = ARCH,
    row_stats: dict[str, pd.Series] | None = None,
    alpha: float = 0.05,
    title: str | None = None,
    save_path: Path | str | None = None,
) -> Figure:
    """Tufte slopegraph: one panel per metric, one sloped line per seed.

    Follows the Beautiful Evidence slopegraph form (Tufte 2006):

    - No grid, no box — all four spines removed; only the data remains.
    - Each seed-pair is a single line; winner seeds rendered in ``palette[arch_a]``
      (solid, full opacity), loser seeds in muted grey.
    - Per-architecture means shown as short horizontal rules flanking the data
      columns, with a numeric value label — marginal notes rather than a legend.
    - Range-frame y-ticks: tick extent clipped to [min(data), max(data)].
    - When ``row_stats`` is supplied, each panel gets a compact stat stamp
      (p, d_z, α) placed below the win-count subtitle. Significant panels are
      rendered in the winner colour; ties in muted grey.

    :param runs: Long-format DataFrame with ``seed_col``, ``arch_col``, and the
                 metric columns listed in ``metrics``.
    :param metrics: List of ``(column_name, panel_title)`` pairs.
    :param arch_a: Architecture plotted on the left (x=0).
    :param arch_b: Architecture plotted on the right (x=1).
    :param seeds: Ordered list of seed identifiers defining the pairing.
    :param palette: ``{arch_label: hex_colour}`` for slope lines and mean rules.
    :param seed_col, arch_col: Column names for seed and architecture.
    :param row_stats: Optional ``{metric_col: Series}`` mapping each metric column
                      to its stats row (from ``build_comparison_table``). Expected
                      keys in each Series: ``wilcoxon_p``, ``cohen_dz``,
                      ``delta_a_minus_b``. Metrics absent from this dict get no stamp.
    :param alpha: Significance threshold used to colour the stat stamp.
    :param title: Figure suptitle. Defaults to ``"Per-seed paired comparison (n=…)"``.
    :param save_path: If given, the figure is written to this path.
    :return: Matplotlib Figure.
    """
    n = len(seeds)
    col_a = palette.get(arch_a, "#c44e58")
    col_b = palette.get(arch_b, "#30638e")
    grey = "#b8b8b8"

    label_a = arch_a.split("+")[-1].strip() if "+" in arch_a else arch_a
    label_b = arch_b.split("+")[-1].strip() if "+" in arch_b else arch_b

    # ── Shared y-scale across all panels (comparable axes) ───────────────────
    all_pivots = {}
    for col, _ in metrics:
        piv = runs.pivot(index=seed_col, columns=arch_col, values=col).loc[seeds]
        all_pivots[col] = piv
    global_vals = pd.concat(
        [pd.concat([p[arch_a], p[arch_b]]) for p in all_pivots.values()]
    )
    y_lo_global = float(global_vals.min())
    y_hi_global = float(global_vals.max())
    pad = (y_hi_global - y_lo_global) * 0.12

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 6.0))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, panel_title) in zip(axes, metrics):
        piv = all_pivots[col]

        for seed in seeds:
            y = [float(piv.loc[seed, arch_a]), float(piv.loc[seed, arch_b])]
            win = y[0] > y[1]
            ax.plot([0, 1], y, "-",
                    color=col_a if win else grey,
                    lw=1.6 if win else 1.0,
                    alpha=1.0 if win else 0.55,
                    solid_capstyle="round",
                    zorder=2)
            ax.scatter([0], [y[0]], color=col_a, s=28, zorder=3, linewidths=0)
            ax.scatter([1], [y[1]], color=col_b, s=28, zorder=3, linewidths=0)

        mean_a = float(piv[arch_a].mean())
        mean_b = float(piv[arch_b].mean())
        ax.plot([-0.06, -0.01], [mean_a, mean_a], color=col_a, lw=2.2, solid_capstyle="butt", zorder=4)
        ax.plot([1.01, 1.06], [mean_b, mean_b], color=col_b, lw=2.2, solid_capstyle="butt", zorder=4)
        ax.annotate(f"{mean_a:.4f}", (-0.07, mean_a),
                    ha="right", va="center", fontsize=9, color=col_a)
        ax.annotate(f"{mean_b:.4f}", (1.07, mean_b),
                    ha="left", va="center", fontsize=9, color=col_b)

        n_a = int((piv[arch_a] > piv[arch_b]).sum())
        ax.set_title(
            f"{panel_title}\n"
            f"{label_a} {n_a}/{n}  ·  {label_b} {n - n_a}/{n}",
            fontsize=12, pad=14,
        )

        # ── stat stamp below win-count — placed below arch labels ─────────────
        if row_stats and col in row_stats:
            sr = row_stats[col]
            p_val = float(sr["wilcoxon_p"])
            dz = float(sr["cohen_dz"])
            delta = float(sr["delta_a_minus_b"])
            sig = p_val < alpha
            winner_col = (col_a if delta > 0 else col_b) if sig else "#999999"
            stamp = f"Wilcoxon p = {p_val:.3f}  ·  Cohen's d_z = {dz:+.2f}  ·  α = {alpha}"
            ax.text(0.5, y_lo_global - pad * 2.5, stamp,
                    ha="center", va="top", fontsize=8.5,
                    color=winner_col, style="italic",
                    transform=ax.transData)

        ax.text(0, y_lo_global - pad * 1.4, label_a, ha="center", va="top",
                fontsize=11, color=col_a, fontweight="semibold")
        ax.text(1, y_lo_global - pad * 1.4, label_b, ha="center", va="top",
                fontsize=11, color=col_b, fontweight="semibold")

        # shared scale — same xlim/ylim on every panel
        ax.set_xlim(-0.38, 1.38)
        ax.set_ylim(y_lo_global - pad * 3.2, y_hi_global + pad * 0.5)
        ax.set_xticks([])
        ax.set_yticks(np.linspace(y_lo_global, y_hi_global, 4))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.tick_params(axis="y", left=True, length=3, width=0.7,
                       labelsize=10, colors="#555555")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.plot([ax.get_xlim()[0]] * 2, [y_lo_global, y_hi_global],
                color="#bbbbbb", lw=0.7, transform=ax.transData, zorder=0, clip_on=False)

    fig.suptitle(title or f"Per-seed paired comparison  (n={n})", fontsize=14, y=1.04)
    fig.tight_layout(w_pad=3.0)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_paired_forest(
    results: pd.DataFrame,
    forest_specs: list[tuple[str, str, bool]],
    *,
    arch_a: str,
    arch_b: str,
    n: int,
    palette: dict[str, str],
    delta_col: str = "delta_a_minus_b",
    ci_lo_col: str = "boot_lo",
    ci_hi_col: str = "boot_hi",
    alpha: float = 0.05,
    title: str | None = None,
    save_path: Path | str | None = None,
) -> Figure:
    """Tufte forest plot of paired A−B differences with BCa bootstrap 95% CIs.

    Follows Beautiful Evidence data-ink principles:

    - Hairline CI bars (lw = 0.9) so confidence intervals read as precision
      information rather than visual mass.
    - Filled dot at the point estimate; size proportional to signal clarity
      so a significant result is visually heavier.
    - Null line (x=0) rendered as a thin mid-grey rule — present but passive.
    - CI bounds annotated as marginal text outside the plot frame (right side)
      so the reader can read exact values without a grid.
    - No grid — the hairline null line provides all necessary orientation.
    - Architecture labels annotated at the extremes of the x-axis instead of
      a colour legend.

    :param results: Indexed by metric name; must contain ``delta_col``,
                    ``ci_lo_col``, and ``ci_hi_col`` columns (the natural output
                    of :func:`~SkiNet.Utils.analysis.stats.paired_metric_stats`).
    :param forest_specs: List of ``(display_label, metric_key, flip_sign)``.
                         ``flip_sign=True`` negates the delta and CI (e.g. for
                         gen-gap where lower is better).
    :param arch_a, arch_b: Architecture labels used for axis annotation.
    :param n: Number of paired seeds (shown in title).
    :param palette: ``{arch_label: hex_colour}``.
    :param delta_col, ci_lo_col, ci_hi_col: Column names in ``results``.
    :param alpha: Significance threshold used to colour the stat stamp on each row.
                  Significant rows are rendered in the winner colour; ties in grey.
    :param title: Axes title. Defaults to a generated bootstrap-CI caption.
    :param save_path: If given, the figure is written to this path.
    :return: Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(9, 0.9 * len(forest_specs) + 1.6))
    ys = list(range(len(forest_specs)))[::-1]

    # null line — thin, grey, passive
    ax.axvline(0, color="#aaaaaa", lw=0.8, zorder=1)

    all_lo, all_hi = [], []
    for y, (label, metric, flip) in zip(ys, forest_specs):
        row = results.loc[metric]
        d = float(row[delta_col])
        lo = float(row[ci_lo_col])
        hi = float(row[ci_hi_col])
        if flip:
            d, lo, hi = -d, -hi, -lo
        all_lo.append(lo)
        all_hi.append(hi)

        # use wilcoxon_p < alpha when available; fall back to CI-excludes-zero
        _p = float(row["wilcoxon_p"]) if "wilcoxon_p" in row.index else None
        significant = (_p < alpha) if _p is not None else (lo > 0 or hi < 0)
        colour = palette.get(arch_a, "#d1495b") if d > 0 else palette.get(arch_b, "#30638e")
        stamp_colour = colour if significant else "#999999"

        # hairline CI bar
        ax.plot([lo, hi], [y, y], color=colour, lw=0.9, zorder=2, solid_capstyle="butt")
        # filled dot — larger when result is significant (data-ink signal)
        ax.scatter([d], [y], color=colour, s=55 if significant else 28,
                   zorder=3, linewidths=0)

        # delta annotation above the dot
        ax.annotate(f"{d:+.4f}", (d, y),
                    xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=colour)

        # marginal right annotations: CI on one line, stats below
        p_val = _p
        dz = float(row["cohen_dz"]) if "cohen_dz" in row.index else None
        x_right = max(all_hi) if all_hi else hi

        ci_text = f"[{lo:+.4f}, {hi:+.4f}]"
        ax.annotate(
            ci_text,
            (x_right, y),
            xytext=(8, 4), textcoords="offset points",
            ha="left", va="bottom", fontsize=9, color=stamp_colour,
            annotation_clip=False,
        )
        stat_parts = []
        if p_val is not None:
            stat_parts.append(f"Wilcoxon p = {p_val:.3f}")
        if dz is not None:
            stat_parts.append(f"d_z = {dz:+.2f}")
        if stat_parts:
            ax.annotate(
                "  ·  ".join(stat_parts),
                (x_right, y),
                xytext=(8, -4), textcoords="offset points",
                ha="left", va="top", fontsize=8.5, color=stamp_colour,
                style="italic", annotation_clip=False,
            )

    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in forest_specs], fontsize=11)
    ax.tick_params(axis="y", length=0)  # no y-axis tick marks — labels are enough

    # architecture labels at the x-axis extremes replace a colour legend
    label_a = arch_a.split("+")[-1].strip() if "+" in arch_a else arch_a
    label_b = arch_b.split("+")[-1].strip() if "+" in arch_b else arch_b
    x_lo = min(all_lo) if all_lo else -0.01
    x_hi = max(all_hi) if all_hi else 0.01
    ax.annotate(f"← {label_b} better", (x_lo, -0.7),
                ha="left", va="top", fontsize=10, color=palette.get(arch_b, "#30638e"),
                annotation_clip=False)
    ax.annotate(f"{label_a} better →", (x_hi, -0.7),
                ha="right", va="top", fontsize=10, color=palette.get(arch_a, "#d1495b"),
                annotation_clip=False)

    ax.set_xlabel("Paired difference  (BCa 95% CI)", fontsize=11)
    ax.set_title(
        title or f"Paired {label_a}−{label_b} differences, bootstrap 95% CI  (n={n})",
        fontsize=12, pad=14,
    )
    # method footnote — Tufte: marginal note rather than in-figure legend
    fig.text(0.01, -0.03,
             "Test: Wilcoxon signed-rank  ·  Effect size: Cohen's d_z  ·  CI: BCa bootstrap 95%",
             ha="left", va="top", fontsize=8, color="#888888", style="italic")
    sns.despine(ax=ax, left=True)
    ax.spines["bottom"].set_color("#aaaaaa")
    ax.spines["bottom"].set_linewidth(0.6)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig

# ---------- E4 threshold-sweep notebooks ------------


def plot_threshold_slopegraph(
    per_seed: pd.DataFrame,
    gain: dict,
    slope_metrics: list[tuple[str, str]],
    *,
    palette: dict[str, str],
    seeds: list[int],
    alpha: float = 0.05,
    n: int | None = None,
    title: str | None = None,
    save_path: "Path | str | None" = None,
) -> Figure:
    """Paired slopegraph for E4: fixed-0.5 → swept-τ* Dice, one line per seed.

    Reshapes ``per_seed`` into the long format ``plot_paired_slopegraph`` expects
    (pseudo-arch column ``tau_condition`` ∈ ``{'swept', 'fixed'}``), attaches the
    gain stats for the Wilcoxon/d_z stamp, and delegates to that function.

    :param per_seed: Output of :func:`~SkiNet.Utils.analysis.threshold_sweep.load_threshold_sweep`.
    :param gain: Output of :func:`~SkiNet.Utils.analysis.threshold_sweep.paired_gain_stats`.
    :param slope_metrics: List of ``(column_name, panel_title)`` pairs, where
                          ``column_name`` is the shared metric column built during
                          the reshape (e.g. ``'val_dice_at_threshold'``).
    :param palette: ``{'swept': hex, 'fixed': hex}``.
    :param seeds: Ordered seed list for pairing.
    :param alpha: Significance threshold for the stat stamp.
    :param n: Number of seeds (inferred from ``seeds`` when ``None``).
    :param title: Figure suptitle.
    :param save_path: If given, the figure is saved here.
    :return: Matplotlib Figure.
    """
    col = slope_metrics[0][0]
    runs_long = pd.concat([
        per_seed[["seed", "val_dice"]].assign(
            tau_condition="fixed", **{col: per_seed["val_dice"]}
        )[["seed", "tau_condition", col]],
        per_seed[["seed", "val_best_dice_at_threshold"]].assign(
            tau_condition="swept", **{col: per_seed["val_best_dice_at_threshold"]}
        )[["seed", "tau_condition", col]],
    ], ignore_index=True)

    row_stats = {col: pd.Series({
        "wilcoxon_p": gain["wilcoxon_p"],
        "cohen_dz": gain["cohen_dz"],
        "delta_a_minus_b": gain["mean"],
    })}

    return plot_paired_slopegraph(
        runs_long, slope_metrics,
        arch_a="swept", arch_b="fixed",
        seeds=seeds, palette=palette,
        seed_col="seed", arch_col="tau_condition",
        row_stats=row_stats, alpha=alpha,
        title=title or f"Fig 1 — Paired Dice: fixed-0.5 vs swept τ* (n={n or len(seeds)})\nin-sample gain — NOT a deployment claim",
        save_path=save_path,
    )


def plot_threshold_forest(
    gain: dict,
    forest_specs: list[tuple[str, str, bool]],
    *,
    palette: dict[str, str],
    n: int,
    alpha: float = 0.05,
    title: str | None = None,
    save_path: "Path | str | None" = None,
) -> Figure:
    """Forest plot for E4: mean Δ (swept − 0.5) with BCa 95% CI.

    Builds the single-row results DataFrame from ``gain`` and delegates to
    :func:`plot_paired_forest`.

    :param gain: Output of :func:`~SkiNet.Utils.analysis.threshold_sweep.paired_gain_stats`.
    :param forest_specs: List of ``(display_label, metric_key, flip_sign)`` — same
                         format as :func:`plot_paired_forest`.
    :param palette: ``{'swept': hex, 'fixed': hex}``.
    :param n: Number of seeds.
    :param alpha: Significance threshold.
    :param title: Axes title.
    :param save_path: If given, the figure is saved here.
    :return: Matplotlib Figure.
    """
    forest_df = pd.DataFrame([{
        "delta_a_minus_b": gain["mean"],
        "boot_lo": gain["ci_lo"],
        "boot_hi": gain["ci_hi"],
        "wilcoxon_p": gain["wilcoxon_p"],
        "cohen_dz": gain["cohen_dz"],
    }], index=[forest_specs[0][1]])

    return plot_paired_forest(
        forest_df, forest_specs,
        arch_a="swept", arch_b="fixed",
        n=n, palette=palette, alpha=alpha,
        title=title or (
            f"Fig 2 — In-sample Dice gain swept−fixed, BCa 95% CI (n={n})\n"
            "(gain is real but in-sample only — see §2.1)"
        ),
        save_path=save_path,
    )


def plot_tau_across_seeds(
    per_seed: pd.DataFrame,
    stab: dict,
    *,
    default_tau: float = 0.5,
    c_fixed: str = "#30638e",
    c_swept: str = "#d1495b",
    c_tau: str = "#3a5a40",
    title: str | None = None,
    save_path: "Path | str | None" = None,
) -> Figure:
    """Scatter of τ* per seed vs the fixed-0.5 reference line (Fig 3).

    :param per_seed: Output of :func:`~SkiNet.Utils.analysis.threshold_sweep.load_threshold_sweep`.
    :param stab: Output of :func:`~SkiNet.Utils.analysis.threshold_sweep.threshold_stability`.
    :param default_tau: Fixed threshold baseline drawn as a dashed reference.
    :param c_fixed, c_swept, c_tau: Colours for the reference line, scatter dots,
                                     and median rule / ±1 SD band.
    :param title: Axes title.
    :param save_path: If given, the figure is saved here.
    :return: Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(default_tau, color=c_fixed, ls="--", lw=1.5, label=f"τ = {default_tau} (incumbent)")
    ax.axhline(stab["tau_median"], color=c_tau, ls="-", lw=1.5,
               label=f"median τ* = {stab['tau_median']:.2f}")
    ax.scatter(per_seed["seed"], per_seed["val_optimal_threshold"], color=c_swept, s=55, zorder=3)
    ax.fill_between(
        per_seed["seed"],
        stab["tau_median"] - stab["tau_sd"],
        stab["tau_median"] + stab["tau_sd"],
        color=c_tau, alpha=0.10,
        label=f"±1 SD = ±{stab['tau_sd']:.2f}",
    )
    ax.set_xlabel("seed")
    ax.set_ylabel("selected threshold τ*")
    ax.set_ylim(0, 1)
    ax.set_xticks(per_seed["seed"])
    ax.set_title(title or (
        f"Fig 3 — τ* across seeds: straddles {default_tau}, no consistent direction\n"
        f"|median − {default_tau}| = {stab['tau_dist_from_half']:.2f}  <  SD = {stab['tau_sd']:.2f}"
    ))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_tau_trajectories(
    traj: dict,
    stab: dict,
    *,
    warmup: int = 50,
    default_tau: float = 0.5,
    tau_key: str = "val_optimal_threshold",
    c_fixed: str = "#30638e",
    title: str | None = None,
    save_path: "Path | str | None" = None,
) -> Figure:
    """Within-run τ* epoch trajectories for all seeds (Fig 4).

    :param traj: Output of :func:`~SkiNet.Utils.analysis.threshold_sweep.epoch_trajectories`,
                 ``{seed: {tau_key: np.ndarray, ...}}``.
    :param stab: Output of :func:`~SkiNet.Utils.analysis.threshold_sweep.threshold_stability`.
    :param warmup: Warm-up cutoff epoch drawn as a vertical reference line.
    :param default_tau: Fixed threshold drawn as a horizontal dashed reference.
    :param tau_key: Key in each ``traj[seed]`` dict holding the τ* array.
    :param c_fixed: Colour for the fixed-threshold reference line.
    :param title: Axes title.
    :param save_path: If given, the figure is saved here.
    :return: Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for seed, d in traj.items():
        ax.plot(np.arange(1, len(d[tau_key]) + 1), d[tau_key], lw=0.8, alpha=0.55)
    ax.axhline(default_tau, color=c_fixed, ls="--", lw=1.5, label=f"τ = {default_tau}")
    ax.axvline(warmup, color="#888888", ls=":", lw=1.0, label=f"warm-up cutoff (ep {warmup})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("argmax threshold τ*")
    ax.set_ylim(0, 1)
    ax.set_title(title or (
        f"Fig 4 — τ* never converges: within-run wander SD ≈ {stab['mean_wander_sd']:.2f}\n"
        f"global range [{stab['tau_global_min']:.2f}, {stab['tau_global_max']:.2f}] — τ* tracks val noise"
    ))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------- E0 batch size sweep notebooks ------------


def scatter_outliers(df_exp: pd.DataFrame,
                     y_col: str,
                     title_prefix: str,
                     ylabel: str,
                     batch_sizes: list[int] | None = None,
                     ylog: bool = False,
                     save_path: "str | Path | None" = None) -> Figure:
    """A scatter plot of grid of y_col vs step per batch size, with outliers highlighted in red.

    Diagnostic view used in the outlier-audit section of batch-size sweep notebooks.
    Clean steps are shown in blue; flagged outliers in red.

    :param df_exp: Tidy DataFrame filtered to a single experiment
                   (columns ``batch_size``, ``step``, ``y_col``, ``is_outlier``).
    :param y_col: Column to plot on the y-axis (e.g. ``"samples_per_sec"``).
    :param title_prefix: Figure suptitle prefix.
    :param ylabel: Y-axis label string.
    :param batch_sizes: Ordered list of batch sizes to use as panel keys.
                        Defaults to ``[4, 8, 16, 32, 64, 128]``.
    :param ylog: If ``True``, use a log scale on the y-axis.
    :param save_path: If given, save the figure to this path as PNG (dpi=150).
    :return: Matplotlib Figure.
    """
    from SkiNet.Utils.analysis.schema import EXPECTED_BATCH_SIZES
    import math
    bs_grid = batch_sizes or EXPECTED_BATCH_SIZES
    n = len(bs_grid)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 7 * nrows / 2), sharey=False)
    axes_flat = list(np.array(axes).flat) if n > 1 else [axes]
    first_ax = axes_flat[0]
    for ax, bs in zip(axes_flat, bs_grid):
        #
        # subset to this batch size
        df_fixed_bs = df_exp[df_exp["batch_size"] == bs]
        if df_fixed_bs.empty:
            ax.set_title(f"bs={bs} (no data)")
            ax.set_xlabel("step")
            ax.set_ylabel(ylabel)
            continue
        # clean vs outlier steps
        df_regular_samples = df_fixed_bs[~df_fixed_bs["is_outlier"]]
        df_outliers = df_fixed_bs[df_fixed_bs["is_outlier"]]
        ax.scatter(df_regular_samples["step"], df_regular_samples[y_col], s=6, c="tab:blue", alpha=0.55, label="clean")
        ax.scatter(df_outliers["step"], df_outliers[y_col], s=14, c="tab:red", alpha=0.85, label="outlier")
        if ylog:
            ax.set_yscale("log")
        ax.set_title(f"bs={bs}")
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        if ax is first_ax:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(title_prefix, y=1.02)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_empirical_throughput_and_timing(summary: pd.DataFrame,
                                         batch_sizes: list[int] | None = None) -> Figure:
    """Dual-panel bar chart: median samples/sec (left) and median ms/step (right).

    Shows raw measured values, where whiskers span the [10th, 90th] percentile range of clean steps.
    Medians are annotated on each bar.

    :param summary: Output of ``throughput_summary`` (with ``p10_samples_per_sec``, ``median_samples_per_sec``,
                    ``p90_samples_per_sec``, ``median_time_per_step_ms`` columns).
    :param batch_sizes: Ordered batch-size grid for the x-axis.
    :return: Matplotlib Figure.
    """
    bs_grid = np.array(sorted(summary["batch_size"].unique()) if batch_sizes is None
                       else batch_sizes)
    experiments = list(summary["experiment"].unique())
    palette = {"no_aug": "tab:blue", "with_aug": "tab:orange"}
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for i, exp in enumerate(experiments):
        g = summary[summary["experiment"] == exp].set_index("batch_size").reindex(bs_grid)
        x = np.arange(len(bs_grid)) + (i - (len(experiments) - 1) / 2) * width
        med = g["median_samples_per_sec"].values
        err_lo = med - g["p10_samples_per_sec"].values
        err_hi = g["p90_samples_per_sec"].values - med
        axes[0].bar(x, med, width=width, yerr=[err_lo, err_hi], capsize=3,
                    label=exp, color=palette.get(exp), alpha=0.85)
        for xi, mi in zip(x, med):
            if np.isfinite(mi):
                axes[0].text(xi, mi, f"{mi:.0f}", ha="center", va="bottom", fontsize=8)

    axes[0].set_xticks(np.arange(len(bs_grid)))
    axes[0].set_xticklabels([str(b) for b in bs_grid])
    axes[0].set_xlabel("batch size")
    axes[0].set_ylabel("samples / sec  (clean median; whiskers = p10–p90)")
    axes[0].set_title("Empirical throughput per batch size")
    axes[0].legend(fontsize=9)

    for i, exp in enumerate(experiments):
        g = summary[summary["experiment"] == exp].set_index("batch_size").reindex(bs_grid)
        x = np.arange(len(bs_grid)) + (i - (len(experiments) - 1) / 2) * width
        tms = g["median_time_per_step_ms"].values
        axes[1].bar(x, tms, width=width, label=exp, color=palette.get(exp), alpha=0.85)
        for xi, ti in zip(x, tms):
            if np.isfinite(ti):
                axes[1].text(xi, ti, f"{ti:.0f}", ha="center", va="bottom", fontsize=8)

    axes[1].set_xticks(np.arange(len(bs_grid)))
    axes[1].set_xticklabels([str(b) for b in bs_grid])
    axes[1].set_xlabel("batch size")
    axes[1].set_ylabel("time / step  [ms]  (clean median)")
    axes[1].set_title("Empirical time per step per batch size")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    return fig


def plot_throughput_traces(df: pd.DataFrame,
                           experiment: str,
                           batch_sizes: list[int] | None = None,
                           max_epochs: int = 10) -> Figure:
    """2×3 grid of samples/sec time series (clean steps only) per batch size.

    Each panel shows the raw trace (blue line), the overall median (dashed),
    and the [p10, p90] band (shaded). Confirms the median is representative
    and that throughput does not drift within a run.

    :param df: Concatenated tidy DataFrame.
    :param experiment: Experiment label to filter on.
    :param batch_sizes: Ordered batch-size grid for panel assignment.
    :param max_epochs: Unused; kept for API symmetry with :func:`plot_loss_curves`.
    :return: Matplotlib Figure.
    """
    from SkiNet.Utils.analysis.schema import EXPECTED_BATCH_SIZES
    bs_grid = batch_sizes or EXPECTED_BATCH_SIZES
    df_e = df[(df["experiment"] == experiment) & (~df["is_outlier"])]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)
    for ax, bs in zip(axes.flat, bs_grid):
        sub = df_e[df_e["batch_size"] == bs]
        if sub.empty:
            ax.set_title(f"bs={bs} (no data)")
            continue
        sub = sub.sort_values("step")
        med = sub["samples_per_sec"].median()
        p10 = sub["samples_per_sec"].quantile(0.10)
        p90 = sub["samples_per_sec"].quantile(0.90)
        ax.plot(sub["step"], sub["samples_per_sec"], color="tab:blue", alpha=0.6, linewidth=0.7)
        ax.axhline(med, color="black", linestyle="--", alpha=0.8, label=f"median={med:.1f}")
        ax.axhspan(p10, p90, color="tab:blue", alpha=0.12, label="p10–p90")
        ax.set_title(f"bs={bs}")
        ax.set_xlabel("step")
        ax.set_ylabel("samples/sec")
        if ax is axes.flat[0]:
            ax.legend(fontsize=8)
    fig.suptitle(f"Throughput stability — experiment: {experiment}", y=1.02)
    fig.tight_layout()
    return fig


def plot_loss_curves(df: pd.DataFrame,
                     experiment: str,
                     batch_sizes: list[int] | None = None,
                     max_epochs: int = 10) -> Figure:
    """2×3 grid of training loss curves per batch size — sanity check only.

    Shows clean step-level loss (blue), outlier steps (grey scatter), and a
    per-epoch mean line (red). Titles flag runs with potential instability
    (finite loss max > 5.0 or non-finite values).

    :param df: Concatenated tidy DataFrame.
    :param experiment: Experiment label to filter on.
    :param batch_sizes: Ordered batch-size grid for panel assignment.
    :param max_epochs: Used to infer steps-per-epoch for the epoch-mean x-positions.
    :return: Matplotlib Figure.
    """
    from SkiNet.Utils.analysis.schema import EXPECTED_BATCH_SIZES
    bs_grid = batch_sizes or EXPECTED_BATCH_SIZES
    df_e = df[df["experiment"] == experiment]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)
    for ax, bs in zip(axes.flat, bs_grid):
        sub = df_e[df_e["batch_size"] == bs]
        if sub.empty or sub["train_loss"].isna().all():
            ax.set_title(f"bs={bs} (no loss data)")
            continue
        sub = sub.sort_values("step")
        clean = sub[~sub["is_outlier"]]
        out = sub[sub["is_outlier"]]
        ax.plot(clean["step"], clean["train_loss"], color="tab:blue", alpha=0.5,
                linewidth=0.6, label="train_loss_step")
        ax.scatter(out["step"], out["train_loss"], s=6, c="lightgray",
                   alpha=0.6, label="outlier step")
        if clean["epoch_idx"].notna().any():
            per_epoch = (
                clean.dropna(subset=["epoch_idx"])
                .groupby(clean["epoch_idx"].astype(int))["train_loss"].mean()
            )
            spe = max(1, sub["step"].nunique() // max_epochs)
            xs = [int((e + 0.5) * spe) for e in per_epoch.index]
            ax.plot(xs, per_epoch.values, color="tab:red", linewidth=2.0,
                    marker="o", label="per-epoch mean")
        finite_loss = clean["train_loss"].replace([np.inf, -np.inf], np.nan).dropna()
        diverged = (not np.isfinite(finite_loss).all()) or (finite_loss.max() > 5.0)
        ax.set_title(f"bs={bs}" + ("  ⚠️ unstable" if diverged else ""))
        ax.set_xlabel("step")
        ax.set_ylabel("train_loss_step")
        if ax is axes.flat[0]:
            ax.legend(fontsize=8)
    fig.suptitle(
        f"Training loss curves — experiment: {experiment}  (sanity check only)", y=1.02
    )
    fig.tight_layout()
    return fig


def plot_gpu_panels(gpu_tbl: pd.DataFrame,
                    batch_sizes: list[int] | None = None) -> Figure:
    """Dual-panel bar chart: GPU utilisation (left) and peak GPU memory (right).

    Draws grouped bars per experiment. Right panel adds reference lines at 16 GB
    (T4 limit) and 12 GB (DDP risk threshold).

    :param gpu_tbl: Output of :func:`~SkiNet.Utils.analysis.batch_sweep.gpu_summary`.
    :param batch_sizes: Ordered batch-size grid. Defaults to
                        :data:`~SkiNet.Utils.analysis.schema.EXPECTED_BATCH_SIZES`.
    :return: Matplotlib Figure.
    """
    from SkiNet.Utils.analysis.schema import EXPECTED_BATCH_SIZES
    bs_grid = np.array(sorted(batch_sizes or EXPECTED_BATCH_SIZES))
    palette = {"no_aug": "tab:blue", "with_aug": "tab:orange"}
    experiments = list(gpu_tbl["experiment"].unique())
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for i, exp in enumerate(experiments):
        g = gpu_tbl[gpu_tbl["experiment"] == exp].set_index("batch_size").reindex(bs_grid)
        offset = (i - (len(experiments) - 1) / 2) * width
        x = np.arange(len(bs_grid)) + offset
        axes[0].bar(x, g["median_gpu_util_pct"].values, width=width,
                    label=exp, color=palette.get(exp), alpha=0.85)
    axes[0].set_xticks(np.arange(len(bs_grid)))
    axes[0].set_xticklabels([str(b) for b in bs_grid])
    axes[0].set_ylim(0, 100)
    axes[0].set_xlabel("batch size")
    axes[0].set_ylabel("median GPU utilisation  [%]")
    axes[0].set_title("GPU utilisation (clean steps)")
    axes[0].legend(fontsize=9)

    for i, exp in enumerate(experiments):
        g = gpu_tbl[gpu_tbl["experiment"] == exp].set_index("batch_size").reindex(bs_grid)
        offset = (i - (len(experiments) - 1) / 2) * width
        x = np.arange(len(bs_grid)) + offset
        axes[1].bar(x, g["peak_gpu_mem_gb"].values, width=width,
                    label=f"{exp} (peak)", color=palette.get(exp), alpha=0.85)
    axes[1].axhline(16, color="red", linestyle="--", alpha=0.7, label="T4 limit (16 GB)")
    axes[1].axhline(12, color="orange", linestyle=":", alpha=0.7, label="DDP risk (12 GB)")
    axes[1].set_xticks(np.arange(len(bs_grid)))
    axes[1].set_xticklabels([str(b) for b in bs_grid])
    axes[1].set_xlabel("batch size")
    axes[1].set_ylabel("peak GPU memory allocated  [GB]")
    axes[1].set_title("Peak GPU memory vs batch size")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    return fig


def plot_sweep_facet(
    df: pd.DataFrame,
    facet_col: str = 'lr',
    facet_order: list[str] | None = None,
    x_col: str = 'samples_per_sec',
    y_col: str = 'val_dice_tail_mean',
    yerr_col: str | None = 'val_dice_tail_std',
    color_col: str = 'merge',
    marker_col: str = 'encoder',
    color_order: list[str] | None = None,
    marker_order: list[str] | None = None,
    color_palette: dict[str, str] | None = None,
    highlight: dict[str, str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = 'Set2',
) -> Figure:
    """One scatter panel per facet value, colour = merge, marker = encoder, ±std bars.

    Splits the sweep DataFrame by ``facet_col`` (e.g. learning rate) and draws
    one axes per value. All panels share the same y-axis scale so Dice levels
    are directly comparable across LRs. Within each panel:

    - **colour** encodes ``color_col`` (default: merge mode) — 3 distinct values,
      making the previously-invisible merge dimension primary.
    - **marker shape** encodes ``marker_col`` (default: encoder family).
    - **vertical error bars** show ``yerr_col`` (default: ``val_dice_tail_std``),
      the convergence noise over the last 10 epochs.
    - Every point is labelled ``encoder+merge`` so the winner is identifiable
      without consulting the table.

    :param df: Sweep DataFrame produced by ``load_sweep_runs``.
    :param facet_col: Column that defines the panels. Defaults to ``'lr'``.
    :param facet_order: Explicit left→right ordering. If ``None``, sorted
                        alphabetically.
    :param x_col: X-axis column. Defaults to ``'samples_per_sec'``.
    :param y_col: Y-axis column (shared scale). Defaults to
                  ``'val_dice_tail_mean'``.
    :param yerr_col: Column used for symmetric ±error bars. Pass ``None`` to
                     suppress. Defaults to ``'val_dice_tail_std'``.
    :param color_col: Column mapped to point colour. Defaults to ``'merge'``.
    :param marker_col: Column mapped to marker shape. Defaults to ``'encoder'``.
    :param color_order: Explicit colour legend ordering.
    :param marker_order: Explicit marker legend ordering.
    :param color_palette: Optional ``{value: hex}`` override for colours.
    :param highlight: Dict of column→value filters plus ``'label'`` key; the
                      matching point gets a ring in every panel where it appears.
    :param xlabel: X-axis label (shared). Defaults to ``x_col``.
    :param ylabel: Y-axis label (left panel only). Defaults to ``y_col``.
    :param figsize: Figure size. Defaults to ``(4 * n_panels, 4.5)``.
    :param cmap: Colormap when ``color_palette`` is not set. Defaults to
                 ``'Set2'`` (qualitative, colourblind-friendly).
    :return: Matplotlib Figure.
    """
    _markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    panels = facet_order or sorted(df[facet_col].unique())
    color_vals = color_order or sorted(df[color_col].unique())
    marker_vals = marker_order or sorted(df[marker_col].unique())
    n = len(panels)

    if color_palette:
        colors = {v: color_palette.get(v, '#888888') for v in color_vals}
    else:
        cmap_obj = cm.get_cmap(cmap)
        colors = {v: mcolors.to_hex(cmap_obj(i / max(len(color_vals) - 1, 1))) for i, v in enumerate(color_vals)}
    marker_map = {v: _markers[i % len(_markers)] for i, v in enumerate(marker_vals)}

    fig, axes = plt.subplots(
        1, n,
        figsize=figsize or (4 * n, 4.5),
        sharey=True,
        constrained_layout=True,
    )
    if n == 1:
        axes = [axes]

    highlight_filters = {}
    highlight_label = ''
    if highlight:
        highlight_filters = {k: v for k, v in highlight.items() if k != 'label'}
        highlight_label = highlight.get('label', '')

    for ax, panel_val in zip(axes, panels):
        sub = df[df[facet_col] == panel_val]

        for _, row in sub.iterrows():
            color = colors.get(row[color_col], '#888888')
            marker = marker_map.get(row[marker_col], 'o')
            yerr = float(row[yerr_col]) if (yerr_col and yerr_col in row.index) else None
            ax.errorbar(
                float(row[x_col]), float(row[y_col]),
                yerr=yerr,
                fmt=marker, color=color,
                markersize=7, alpha=0.9,
                elinewidth=1.2, capsize=3, capthick=1.2,
                zorder=3,
            )
            ax.annotate(
                f"{row['encoder']}\n+{row['merge']}",
                (float(row[x_col]), float(row[y_col])),
                xytext=(5, 3), textcoords='offset points',
                fontsize=6, color='#333333',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.55),
            )

        # ring + bold label for highlighted run
        if highlight_filters:
            mask = pd.Series(True, index=sub.index)
            for col, val in highlight_filters.items():
                if col in sub.columns:
                    mask &= sub[col] == val
            for _, row in sub[mask].iterrows():
                ax.scatter(row[x_col], row[y_col], s=260, facecolors='none',
                           edgecolors='black', linewidths=2.2, zorder=5)
                if highlight_label:
                    ax.annotate(highlight_label,
                                (float(row[x_col]), float(row[y_col])),
                                xytext=(6, 8), textcoords='offset points',
                                fontsize=7, fontweight='bold', color='black')

        ax.set_title(f'{facet_col}={panel_val}', fontsize=9, fontweight='bold')
        ax.set_xlabel(xlabel or x_col, fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel(ylabel or y_col, fontsize=8)

    # shared legend below all panels
    color_handles = [
        mpatches.Patch(color=colors[v], label=f'{color_col}={v}') for v in color_vals
    ]
    marker_handles = [
        mlines.Line2D([], [], color='#555555', marker=marker_map[v],
                      linestyle='None', markersize=6, label=f'{marker_col}={v}')
        for v in marker_vals
    ]
    fig.legend(handles=color_handles + marker_handles,
               loc='lower center', ncol=len(color_vals) + len(marker_vals),
               fontsize=7, bbox_to_anchor=(0.5, -0.08), frameon=False)
    return fig
