"""Paired-comparison statistics for segmentation benchmark analysis.

All public functions operate on a 1-D array of *paired differences*
``d[i] = A_i - B_i`` (one element per shared seed).

The primary test os Wilcoxon signed-rank, the primary CI method is BCa (bias-corrected and accelerated) bootstrap via
``scipy.stats.bootstrap`` [1], which corrects for estimator bias and skew and
is distribution-free — preferred over the plain percentile method at small n.
Cohen's d_z [2] is the paired effect size (paired denominator, not pooled SD).
Statistical recommendations follow Rainio et al. [3].

References
----------
[1] B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*,
    Chapman & Hall/CRC, Boca Raton, FL, USA (1993).
[2] J. Cohen, *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.),
    Lawrence Erlbaum Associates, Hillsdale, NJ, USA (1988).
[3] O. Rainio, J. Teuho, R. Klén, "Evaluation metrics and statistical tests
    for machine learning", *Scientific Reports* 14, 6086 (2024).
    https://doi.org/10.1038/s41598-024-54515-w
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Callable

import numpy as np
import pandas as pd
from scipy.stats import bootstrap as _scipy_bootstrap
from scipy.stats import binomtest, wilcoxon

from SkiNet.Utils.analysis.schema import ARCH, SEED


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cohen_dz_scalar(x: np.ndarray) -> float:
    """Cohen's d_z = mean(x) / SD(x, ddof=1)."""
    s = float(x.std(ddof=1))
    return float(x.mean()) / s if s > 0 else float("nan")


def _bca_ci(
    d: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_resamples: int,
    confidence_level: float,
    random_state: int | np.random.Generator,
) -> tuple[float, float]:
    """Run BCa bootstrap and raise if the result contains NaN.

    BCa returns NaN CI endpoints when the bootstrap distribution contains
    enough NaN resamples to make the bias-correction formula undefined.
    For d_z this can occur when a degenerate resample has SD = 0.
    Rather than silently falling back to a non-BCa method, we raise so the
    caller knows the BCa guarantee no longer holds.
    """
    res = _scipy_bootstrap(
        (d,),
        statistic,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="BCa",
        random_state=random_state,
    )
    lo = float(res.confidence_interval.low)
    hi = float(res.confidence_interval.high)

    if np.isnan(lo) or np.isnan(hi):
        n_nan = int(np.isnan(res.bootstrap_distribution).sum())
        raise ValueError(
            f"BCa CI is NaN: {n_nan}/{n_resamples} bootstrap resamples returned NaN "
            f"(likely degenerate resamples with SD=0). "
            f"Increase n or inspect the difference vector d."
        )

    return lo, hi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bootstrap_paired_ci(d: np.ndarray,
                        *,
                        n_resamples: int = 10_000,
                        confidence_level: float = 0.95,
                        random_state: int | np.random.Generator = 42) -> tuple[float, float]:
    """BCa bootstrap CI for mean(Δ).

    The interval comes from :func:`_bca_ci`, the shared BCa engine.

    Parameters
    ----------
    d:
        1-D array of n paired differences Δᵢ = A_i − B_i.
    n_resamples:
        Number of bootstrap resamples. Default 10 000 stabilises the 2.5/97.5
        tail quantiles to ≈ ±0.002 in probability units.
    confidence_level:
        Nominal coverage. Default 0.95.
    random_state:
        Integer seed or ``numpy.random.Generator`` for reproducibility.

    Returns
    -------
    mean_ci:
        ``(low, high)`` BCa CI for mean(Δ).

    Raises
    ------
    ValueError
        If BCa cannot be computed.
    """
    d = np.asarray(d, dtype=float)
    return _bca_ci(d, np.mean,
                   n_resamples=n_resamples,
                   confidence_level=confidence_level,
                   random_state=random_state)


# ---------------------------------------------------------------------------
# Paired metric comparison (A vs B across shared seeds)
# ---------------------------------------------------------------------------

def paired_metric_stats(long_df: pd.DataFrame,
                        metric: str,
                        *,
                        arch_a: str,
                        arch_b: str,
                        seeds: Sequence[int] | None = None,
                        higher_is_better: bool = True,
                        alpha: float = 0.05,
                        n_corrections: int = 1,
                        n_resamples: int = 10_000,
                        random_state: int | np.random.Generator = 42) -> dict[str, float | bool | str]:
    """Full paired A−B comparison for a single metric across shared seeds.

    Pivots ``long_df`` to one paired difference per seed, ``Δᵢ = A_i − B_i``,
    then reports the primary test (Wilcoxon signed-rank), the primary CI (BCa
    bootstrap on mean Δ), Cohen's d_z point estimate, and the sign test.
    Statistical choices follow Rainio et al. [3].

    Parameters
    ----------
    long_df:
        Long-format runs table with one row per (seed, arch). Must contain the
        :data:`~SkiNet.Utils.analysis.schema.SEED`,
        :data:`~SkiNet.Utils.analysis.schema.ARCH`, and ``metric`` columns.
    metric:
        Column to compare.
    arch_a, arch_b:
        Architecture labels; the reported difference is ``A − B``.
    seeds:
        Seeds to include (defines pairing order). Defaults to the sorted set of
        seeds present for both architectures.
    higher_is_better:
        Direction used only for the sign-test win count. ``True`` counts seeds
        where A > B; ``False`` counts A < B. The signed Δ and all CIs/p-values
        are direction-agnostic.
    alpha:
        Family-wise significance level. The per-metric Bonferroni threshold is
        ``alpha / n_corrections``.
    n_corrections:
        Size of the multiplicity family (k). Used only to compute the
        Bonferroni-adjusted threshold flag.
    n_resamples, random_state:
        Bootstrap settings, forwarded to :func:`bootstrap_paired_ci`.

    Returns
    -------
    dict
        One row of statistics, keyed for direct assembly into a DataFrame:
        means, signed Δ, bootstrap CI, Wilcoxon p + Bonferroni flag, Cohen's
        d_z, and sign-test count/p.

    Raises
    ------
    KeyError
        If ``long_df`` is missing any of the ``SEED``, ``ARCH``, or ``metric``
        columns.
    """
    missing = {SEED, ARCH, metric} - set(long_df.columns)
    if missing:
        raise KeyError(
            f"long_df is missing required column(s): {sorted(missing)}. "
            f"Present columns: {sorted(long_df.columns)}"
        )

    # pivot the df to have one row per seed
    piv = long_df.pivot(index=SEED, columns=ARCH, values=metric)
    if seeds is not None:
        piv = piv.loc[list(seeds)]

    # pull out metrics for A anbd B archiectures; the pivot ensures they are aligned by seed
    a, b = piv[arch_a], piv[arch_b]
    d = (a - b).to_numpy()
    if np.isnan(d).any():
        raise ValueError(
            f"Found {int(np.isnan(d).sum())} unpaired seed(s): a seed is present "
            f"for only one of {arch_a!r}/{arch_b!r}. Pass `seeds=` with the shared "
            f"set or drop the offending rows."
        )
    n = d.size

    # precompute mean, SD
    md, sd = float(d.mean()), float(d.std(ddof=1))

    # Bootstrap CI for mean(Δ)
    boot_lo, boot_hi = bootstrap_paired_ci(d, n_resamples=n_resamples, random_state=random_state)
    dz = md / sd if sd > 0 else float("nan")

    # Wilcoxon signed-rank test
    wilcoxon_p = float(wilcoxon(a, b).pvalue)

    # Sign test: count seeds where A > B (or A < B if higher_is_better=False) and compute two-sided p-value.
    a_wins = int((d > 0).sum() if higher_is_better else (d < 0).sum())
    sign_p = float(binomtest(a_wins, n, 0.5, alternative="two-sided").pvalue)

    # Bonferroni correction flag for the Wilcoxon test
    alpha_bonf = alpha / n_corrections

    return {
        "metric": metric,
        "a_mean": float(a.mean()),
        "b_mean": float(b.mean()),
        "delta_a_minus_b": md,
        "boot_lo": boot_lo,
        "boot_hi": boot_hi,
        "wilcoxon_p": wilcoxon_p,
        "bonferroni_sig": wilcoxon_p < alpha_bonf,
        "cohen_dz": dz,
        "sign_n_a": a_wins,
        "sign_p_2tail": sign_p,
    }


def build_comparison_table(long_df: pd.DataFrame,
                           metrics_spec: list[tuple[str, bool, int]],
                           *,
                           arch_a: str,
                           arch_b: str,
                           seeds: Sequence[int] | None = None,
                           alpha: float = 0.05,
                           n_resamples: int = 10_000,
                           random_state: int | np.random.Generator = 42) -> pd.DataFrame:
    """Run paired A−B statistics for a list of metrics and return a combined table.

    Parameters
    ----------
    long_df:
        Long-format runs table (see :func:`paired_metric_stats`).
    metrics_spec:
        Ordered list of ``(metric, higher_is_better, n_corrections)`` triples.
        ``n_corrections`` is the family size k passed to
        :func:`paired_metric_stats` for the Bonferroni flag on that row.
    arch_a, arch_b, seeds, alpha, n_resamples, random_state:
        Forwarded to :func:`paired_metric_stats`.

    Returns
    -------
    pandas.DataFrame
        One row per metric, indexed by ``metric``, with generic column names
        (``a_mean``, ``b_mean``, ``delta_a_minus_b``, …).  Rename columns in
        the caller if arch-specific labels are needed for display.
    """
    rows = [
        paired_metric_stats(
            long_df,
            metric,
            arch_a=arch_a,
            arch_b=arch_b,
            seeds=seeds,
            higher_is_better=higher_is_better,
            alpha=alpha,
            n_corrections=n_corrections,
            n_resamples=n_resamples,
            random_state=random_state,
        )
        for metric, higher_is_better, n_corrections in metrics_spec
    ]
    return pd.DataFrame(rows).set_index("metric")


def holm_step_down(
    pvalues: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Holm step-down multiple-comparison correction.

    Sorts the p-values ascending and rejects H₀ for the i-th smallest (0-based)
    while ``p < alpha / (k − i)``, stopping at the first failure (step-down
    monotonicity). Controls the family-wise error rate and is uniformly more
    powerful than Bonferroni.

    Parameters
    ----------
    pvalues:
        Mapping of test name → p-value (the multiplicity family).
    alpha:
        Family-wise error rate. Default 0.05.

    Returns
    -------
    pandas.DataFrame
        Indexed by test name in ascending-p order with columns
        ``p``, ``threshold``, and ``reject`` (bool).
    """
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    k = len(ordered)
    rows, still_rejecting = [], True
    for i, (name, p) in enumerate(ordered):
        threshold = alpha / (k - i)
        still_rejecting = still_rejecting and (p < threshold)
        rows.append({"test": name, "p": float(p),
                     "threshold": threshold, "reject": still_rejecting})
    return pd.DataFrame(rows).set_index("test")
