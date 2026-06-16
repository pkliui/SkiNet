"""Tabular display and text-verdict reporting for paired-architecture comparisons,
batch-size sweep recommendations, and test-set threshold calibration (EF)."""

from __future__ import annotations

from typing import Sequence
import pandas as pd
from IPython.display import display

from SkiNet.Utils.analysis.schema import ARCH, SEED
from SkiNet.Utils.analysis.stats import holm_step_down

_DEFAULT_RUN_COLS: tuple[str, ...] = (
    ARCH, SEED,
    "val_dice_max", "val_dice_tail_mean", "val_dice_tail_std",
    "val_iou_max", "generalization_gap_final",
    "samples_per_sec", "duration_min",
)


def show_run_table(runs: pd.DataFrame,
                   columns: Sequence[str] | None = None,
                   *,
                   sort_by: Sequence[str] = (ARCH, SEED)) -> None:
    """
    Display a per-run table restricted to ``columns`` and sorted by ``sort_by``,
    so notebooks can show the raw per-(seed, arch) metrics.

    :param runs: Long-format runs table (one row per seed × architecture).
    :param columns: Columns to show, in display order.
                    Defaults to :data:`_DEFAULT_RUN_COLS`.
    :param sort_by: Columns to sort rows by. Default ``("arch", "seed")``.
    """
    if columns is None:
        columns = [c for c in _DEFAULT_RUN_COLS if c in runs.columns]
    display(runs[list(columns)].sort_values(list(sort_by)).reset_index(drop=True))


def _display_paired_table(
    rows: list[dict],
    index: list[str],
    *,
    label_a: str,
    label_b: str,
    delta_label: str,
    extra_cols: dict | None = None,
) -> None:
    """Render a formatted paired-comparison table.

    Shared rendering engine for :func:`show_comparison_table` and
    :func:`show_gain_table`.  Callers prepare a list of row dicts with keys
    ``a_mean``, ``b_mean``, ``delta``, ``ci_lo``, ``ci_hi``, ``wilcoxon_p``,
    ``sig``, ``cohen_dz``; any additional columns are appended via
    ``extra_cols`` (``{col_label: [values]}``) in display order.

    Parameters
    ----------
    rows:
        One dict per table row with keys listed above.
    index:
        Row index labels in the same order as ``rows``.
    label_a, label_b:
        Display-name headers for the two conditions.
    delta_label:
        Header for the Δ column (e.g. ``"Δ (AG−HE2)"``).
    extra_cols:
        Optional ordered dict of ``{col_header: [values]}`` appended after d_z.
    """
    def _ci(lo: float, hi: float) -> str:
        return f"[{lo:+.4f}, {hi:+.4f}]"
    data: dict = {
        label_a: [round(r["a_mean"], 4) for r in rows],
        label_b: [round(r["b_mean"], 4) for r in rows],
        delta_label: [f"{r['delta']:+.4f}" for r in rows],
        "95% BCa CI": [_ci(r["ci_lo"], r["ci_hi"]) for r in rows],
        "wilcoxon_p": [f"{r['wilcoxon_p']:.4f}" for r in rows],
        "sig": [r["sig"] for r in rows],
        "d_z": [round(r["cohen_dz"], 2) for r in rows],
    }
    if extra_cols:
        data.update(extra_cols)
    display(pd.DataFrame(data, index=index))


def show_comparison_table(results: pd.DataFrame) -> None:
    """Display a compact paired-comparison table from :func:`build_comparison_table` output.

    Formats Δ and BCa CI as signed strings; collapses the two CI columns into
    one; reduces to the columns readable at a glance.

    Parameters
    ----------
    results:
        DataFrame produced by :func:`build_comparison_table` (generic column
        names: ``a_mean``, ``b_mean``, ``delta_a_minus_b``, ``boot_lo``,
        ``boot_hi``, ``wilcoxon_p``, ``bonferroni_sig``, ``cohen_dz``).
    """
    rows = [
        {
            "a_mean": r["a_mean"],
            "b_mean": r["b_mean"],
            "delta": r["delta_a_minus_b"],
            "ci_lo": r["boot_lo"],
            "ci_hi": r["boot_hi"],
            "wilcoxon_p": r["wilcoxon_p"],
            "sig": "✓" if r["bonferroni_sig"] else "",
            "cohen_dz": r["cohen_dz"],
        }
        for _, r in results.iterrows()
    ]
    _display_paired_table(
        rows, list(results.index),
        label_a="AG", label_b="HE2",
        delta_label="Δ (AG−HE2)",
    )


def show_gain_table(
    per_seed: pd.DataFrame,
    gain: dict,
    *,
    alpha: float = 0.05,
) -> None:
    """
    Display the paired in-sample gain table for a threshold sweep and print the H₀ verdict.

    Mirrors the column layout of :func:`show_comparison_table` for the
    threshold-sweep context: one row summarising the swept-τ* vs fixed-0.5
    comparison, followed by a per-seed Δ listing and the Wilcoxon decision.

    Parameters
    ----------
    per_seed:
        Output of :func:`~SkiNet.Utils.analysis.threshold_sweep.load_threshold_sweep`.
    gain:
        Output of :func:`~SkiNet.Utils.analysis.threshold_sweep.paired_gain_stats`.
    alpha:
        Significance threshold for the H₀ rejection verdict. Default 0.05.
    """
    rows = [{
        "a_mean": per_seed["val_best_dice_at_threshold"].mean(),
        "b_mean": per_seed["val_dice"].mean(),
        "delta": gain["mean"],
        "ci_lo": gain["ci_lo"],
        "ci_hi": gain["ci_hi"],
        "wilcoxon_p": gain["wilcoxon_p"],
        "sig": "✓" if gain["wilcoxon_p"] < alpha else "",
        "cohen_dz": gain["cohen_dz"],
    }]
    _display_paired_table(
        rows, ["in_sample_dice_gain"],
        label_a="val_best_dice_at_threshold",
        label_b="val_dice (τ=0.5)",
        delta_label="Δ (swept−0.5)",
        extra_cols={"wins": [f"{gain['n_positive']}/{gain['n']}"]},
    )
    reject = gain["wilcoxon_p"] < alpha and not (gain["ci_lo"] <= 0 <= gain["ci_hi"])
    print(f"  per-seed Δ : {gain['per_seed_val_dice_gain']}")
    print(f"  → {'REJECT H0 (gain ≠ 0)' if reject else 'fail to reject H0'} — in-sample only")


def show_family_verdicts(
    results: pd.DataFrame,
    primary: str,
    secondary: Sequence[str],
    *,
    alpha: float = 0.05,
    throughput_metric: str = "samples_per_sec",
) -> None:
    """Print H₀ rejection verdicts for each hypothesis family.

    Covers three independent families: the pre-registered primary (standalone,
    k=1), the secondary quality family (Holm step-down), and the training-cost
    hypothesis (standalone, k=1).

    Parameters
    ----------
    results:
        DataFrame produced by :func:`build_comparison_table`.
    primary:
        Name of the pre-registered primary metric (index label in ``results``).
    secondary:
        Ordered sequence of secondary metric names for Holm correction.
    alpha:
        Family-wise error rate. Default 0.05.
    throughput_metric:
        Index label of the training-cost metric. Default ``"samples_per_sec"``.
    """
    k = len(secondary)

    p_primary = results.loc[primary, "wilcoxon_p"]
    verdict = "REJECT H0 - OK" if p_primary < alpha else "retain H0"
    print(f"Primary  {primary}  (k=1, α={alpha}):")
    print(f"  p={p_primary:.4f}  →  {verdict}")

    holm = holm_step_down(results.loc[list(secondary), "wilcoxon_p"].to_dict(), alpha=alpha)
    print(f"\nHolm step-down  secondary family  (k={k}, α_adj={alpha/k:.4f}):")
    print(holm[["p", "threshold", "reject"]].to_string())

    p_tput = results.loc[throughput_metric, "wilcoxon_p"]
    verdict_tput = "REJECT H0 - OK" if p_tput < alpha else "retain H0"
    print(f"\nThroughput  {throughput_metric}  (k=1, α={alpha}):")
    print(f"  p={p_tput:.4f}  →  {verdict_tput}")
