"""Matplotlib plotting helpers for MLflow experiment visualisation."""

from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


# ── Shared colour helper ──────────────────────────────────────────────────────

def _tail_mean_palette(
    run_summary: pd.DataFrame,
    monitor: str = "val_dice",
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
    """Plot encoder × merge heatmap for a run-level metric.

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
    run_summary: pd.DataFrame | None = None,
    monitor: str = "val_dice",
) -> Figure:
    """Plot one metric history curve per architecture, coloured by tail-mean rank.

    :param history: Long-format DataFrame with columns ``architecture``, ``key``,
                    ``epoch``, ``value``.
    :param key: Metric name to filter on (e.g. ``"val_dice"``).
    :param ylabel: Y-axis label.
    :param title: Axes title.
    :param run_summary: If provided, line colours follow the RdYlGn scale ranked
                        by tail-mean Dice (best = green, worst = red).
    :param monitor: Base metric name used to derive the tail-mean column when
                    ``run_summary`` is given.
    :return: Matplotlib Figure.
    """
    colour_map = _tail_mean_palette(run_summary, monitor) if run_summary is not None else {}

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
    title: str = "Train vs validation Dice by epoch",
    run_summary: pd.DataFrame | None = None,
    monitor: str = "val_dice",
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


def plot_generalization_gap(run_summary: pd.DataFrame) -> Figure:
    """Plot final train-validation Dice gap for each architecture.

    :param run_summary: DataFrame with columns ``encoder``, ``merge``,
                        ``best_val_best_dice_at_threshold`` (used for sort order),
                        and ``generalization_gap_final`` (train Dice − val Dice).
    :return: Matplotlib Figure.
    """
    sort_col = next((c for c in ("val_dice_max", "val_dice") if c in run_summary.columns), run_summary.columns[0])
    plot_df = run_summary.sort_values(sort_col)
    labels = plot_df["encoder"] + " + " + plot_df["merge"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(labels, plot_df["generalization_gap_final"], color="#d55e00")
    ax.set_title("Final train-validation Dice gap")
    ax.set_xlabel("train Dice - val Dice")
    ax.set_ylabel("")
    for idx, value in enumerate(plot_df["generalization_gap_final"]):
        ax.text(value + 0.001, idx, f"{value:.3f}", va="center")
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
    monitor: str = "val_dice",
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


def plot_sweep_scatter(
    df: pd.DataFrame,
    x_col: str = 'samples_per_sec',
    y_col: str = 'val_dice_tail_mean',
    color_col: str = 'lr',
    marker_col: str = 'encoder',
    color_order: list[str] | None = None,
    marker_order: list[str] | None = None,
    highlight: dict[str, str] | None = None,
    label_arch_cols: list[str] | None = None,
    color_palette: dict[str, str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (9, 6),
    cmap: str = 'cividis',
) -> Figure:
    """Scatter plot with dual visual encoding: colour = one sweep axis, marker = another.

    This is the accuracy-vs-throughput view for sweep DataFrames where two
    categorical dimensions need to be distinguished simultaneously. Colour
    encodes ``color_col`` (e.g. learning rate) using a sequential palette
    sampled from ``cmap``; marker shape encodes ``marker_col`` (e.g. encoder
    family). Together they let you read off, for any point, both its training
    cost and which design choices produced it.

    Use this graph when you want to answer: *"do any architectures dominate —
    high Dice AND high throughput — regardless of LR, or does the Pareto front
    shift with LR?"* Points in the top-right corner are unambiguously better;
    points that are fast but low-Dice or accurate but slow require a deliberate
    cost/quality trade-off.

    Pass ``highlight`` to annotate a specific run with a label and ring, e.g.::

        highlight={'encoder': 'classical', 'merge': 'he2', 'lr': '3e-4',
                   'label': '★ classical+he2 (recommended)'}

    Pass ``color_palette`` to override the cmap sampling with explicit colours::

        color_palette={'1e-4': '#003f88', '3e-4': '#00b4d8',
                       '6e-4': '#f77f00', '1e-3': '#d62828'}

    Pass ``label_arch_cols`` to annotate each unique architecture with its name,
    placed at its best (highest ``y_col``) run so labels don't stack::

        label_arch_cols=['encoder', 'merge']   # labels as "classical+he2" etc.

    :param df: Sweep DataFrame with ``x_col``, ``y_col``, ``color_col``, and
               ``marker_col`` columns.
    :param x_col: Column for the x-axis. Defaults to ``'samples_per_sec'``.
    :param y_col: Column for the y-axis. Defaults to ``'val_dice_tail_mean'``.
    :param color_col: Column whose values map to colours (e.g. ``'lr'``).
    :param marker_col: Column whose values map to marker shapes (e.g. ``'encoder'``).
    :param color_order: Explicit ordering for colour legend entries. If ``None``,
                        sorted alphabetically.
    :param marker_order: Explicit ordering for marker legend entries. If ``None``,
                         sorted alphabetically.
    :param highlight: Optional dict of column→value filters identifying one run
                      to annotate with a ring, plus a ``'label'`` key for the text.
    :param label_arch_cols: Columns to combine into a per-architecture label
                            (e.g. ``['encoder', 'merge']``). One label is drawn
                            per unique combination, at that architecture's highest
                            ``y_col`` point, so labels never stack across LRs.
    :param color_palette: Optional ``{value: hex}`` dict that overrides ``cmap``
                          sampling entirely.
    :param title: Axes title. Defaults to a generated string.
    :param xlabel: X-axis label. Defaults to ``x_col``.
    :param ylabel: Y-axis label. Defaults to ``y_col``.
    :param figsize: Figure size in inches.
    :param cmap: Sequential colormap for the colour dimension when ``color_palette``
                 is not provided. Defaults to ``'cividis'``.
    :return: Matplotlib Figure.
    """
    _markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    color_vals = color_order or sorted(df[color_col].unique())
    marker_vals = marker_order or sorted(df[marker_col].unique())

    if color_palette:
        colors = {v: color_palette.get(v, '#888888') for v in color_vals}
    else:
        colour_map = cm.get_cmap(cmap)
        colors = {
            v: mcolors.to_hex(colour_map(i / max(len(color_vals) - 1, 1)))
            for i, v in enumerate(color_vals)
        }
    marker_map = {v: _markers[i % len(_markers)] for i, v in enumerate(marker_vals)}

    fig, ax = plt.subplots(figsize=figsize)
    for _, row in df.iterrows():
        ax.scatter(
            row[x_col], row[y_col],
            color=colors.get(row[color_col], '#888888'),
            marker=marker_map.get(row[marker_col], 'o'),
            s=70, alpha=0.85,
        )

    # highlight a key finding with a ring and label
    if highlight:
        label = highlight.pop('label', '')
        mask = pd.Series(True, index=df.index)
        for col, val in highlight.items():
            mask &= df[col] == val
        for _, row in df[mask].iterrows():
            ax.scatter(row[x_col], row[y_col], s=220, facecolors='none',
                       edgecolors='black', linewidths=2.0, zorder=5)
            ax.annotate(label, (float(row[x_col]), float(row[y_col])),
                        xytext=(8, 6), textcoords='offset points',
                        fontsize=8, fontweight='bold', color='black')

    # one label per unique architecture at its best point
    if label_arch_cols:
        best_idx = df.groupby(label_arch_cols)[y_col].idxmax()
        for idx in best_idx:
            row = df.loc[idx]
            label_text = '+'.join(str(row[c]) for c in label_arch_cols)
            ax.annotate(
                label_text,
                (float(row[x_col]), float(row[y_col])),
                xytext=(5, 4), textcoords='offset points',
                fontsize=7, color='#222222',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6),
            )

    color_handles = [
        mpatches.Patch(color=colors[v], label=f'{color_col}={v}')
        for v in color_vals
    ]
    marker_handles = [
        mlines.Line2D([], [], color='#555555', marker=marker_map[v],
                      linestyle='None', label=f'{marker_col}={v}')
        for v in marker_vals
    ]
    ax.legend(handles=color_handles + marker_handles, loc='lower right',
              fontsize=8, ncol=2)

    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.set_title(title or f'{y_col} vs {x_col}  [{color_col}=colour, {marker_col}=marker]')
    ax.grid(alpha=0.3)
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
