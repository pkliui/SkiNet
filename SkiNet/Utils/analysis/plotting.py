"""Matplotlib plotting helpers for MLflow experiment visualisation."""

from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


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
    image = ax.imshow(heat.values, cmap="viridis")
    ax.set_xticks(range(len(heat.columns)), heat.columns)
    ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_xlabel("Merge residual mode")
    ax.set_ylabel("Encoder residual mode")
    ax.set_title(title)
    for row_idx in range(heat.shape[0]):
        for col_idx in range(heat.shape[1]):
            val = heat.iloc[row_idx, col_idx]
            label = f"{val:.4f}" if pd.notna(val) else "n/a"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label=value_col)
    fig.tight_layout()
    return fig


def plot_learning_curves(history: pd.DataFrame, key: str, ylabel: str, title: str) -> Figure:
    """Plot one metric history curve per architecture.

    :param history: Long-format DataFrame with columns ``architecture``, ``key``,
                    ``epoch``, ``value``.  If ``key`` matches no rows the figure
                    is returned with empty axes.
    :param key: Metric name to filter on (e.g. ``"val_dice"``).
    :param ylabel: Y-axis label.
    :param title: Axes title.
    :return: Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for architecture, sub in history[history["key"].eq(key)].groupby("architecture"):
        ax.plot(sub["epoch"], sub["value"], label=architecture, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_train_val_overlay(history: pd.DataFrame, title: str = "Train vs validation Dice by epoch") -> Figure:
    """Plot train Dice (dashed) and val Dice (solid) together per architecture.

    Colour encodes architecture; line style encodes train vs validation.  This
    shows *when* the train/val gap opens and whether it keeps growing, which is
    more informative than a single final-epoch bar.

    Expects ``history`` to contain rows with ``key`` values ``"train_dice"`` and
    ``"val_dice"``.  Architectures with no data for a key are silently skipped.

    :param history: Output of ``epoch_metrics`` (long-format with columns
                    ``architecture``, ``key``, ``epoch``, ``value``).
    :param title: Axes title.
    :return: Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5.2))
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    architectures = sorted(history["architecture"].unique())

    for idx, architecture in enumerate(architectures):
        color = palette[idx % len(palette)]
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
    plot_df = run_summary.sort_values("best_val_best_dice_at_threshold")
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


def plot_accuracy_throughput(
    run_summary: pd.DataFrame,
    monitor: str = "val_best_dice_at_threshold",
) -> Figure:
    """Plot accuracy-throughput frontier with peak (hollow) and tail mean ± std (filled) overlaid.

    Column names are derived from ``monitor``:

    * ``best_{monitor}`` — required; peak value per run.
    * ``best_{monitor}_tail_mean`` — optional; tail-epoch mean.
    * ``best_{monitor}_tail_std`` — optional; used as error-bar half-width when present.

    :param run_summary: DataFrame with columns ``encoder``, ``merge``,
                        ``samples_per_sec``, and the derived columns above.
    :param monitor: Base metric name used to construct column lookups.
    :return: Matplotlib Figure.
    """
    best_col = f"best_{monitor}"
    tail_col = f"best_{monitor}_tail_mean"
    std_col = f"best_{monitor}_tail_std"
    has_tail = tail_col in run_summary.columns
    has_std = std_col in run_summary.columns

    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_handles, color_labels = [], []

    for idx, (merge, sub) in enumerate(run_summary.groupby("merge")):
        color = palette[idx % len(palette)]
        # peak — hollow markers
        sc = ax.scatter(sub["samples_per_sec"], sub[best_col], s=100,
                        facecolors="none", edgecolors=color, linewidths=1.8,
                        label=merge, zorder=3)
        color_handles.append(sc)
        color_labels.append(merge)

        for _, row in sub.iterrows():
            ax.annotate(str(row["encoder"]), (float(row["samples_per_sec"]), float(row[best_col])),
                        xytext=(5, 4), textcoords="offset points", fontsize=8, color=color)

        if has_tail:
            # tail mean — filled markers with optional error bars
            ax.scatter(sub["samples_per_sec"], sub[tail_col], s=100,
                       color=color, zorder=4)
            if has_std:
                ax.errorbar(sub["samples_per_sec"], sub[tail_col], yerr=sub[std_col],
                            fmt="none", ecolor=color, capsize=4, linewidth=1.2, zorder=2)

    # marker-style legend entries
    peak_handle = mlines.Line2D([], [], marker="o", linestyle="none", markersize=8,
                                markerfacecolor="none", markeredgecolor="grey", label="Peak")
    tail_handle = mlines.Line2D([], [], marker="o", linestyle="none", markersize=8,
                                color="grey", label="Tail mean ± std")

    first_legend = ax.legend(color_handles, color_labels, title="Merge mode",
                             loc="lower right")
    ax.add_artist(first_legend)
    ax.legend(handles=[peak_handle, tail_handle], loc="upper left")

    ax.set_title("Accuracy-throughput frontier")
    ax.set_xlabel("Final training throughput, samples/sec")
    ax.set_ylabel("Validation Dice-at-threshold")
    fig.tight_layout()
    return fig
