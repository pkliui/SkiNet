"""Per-epoch sigmoid-threshold sweep analysis (experiment E4).

When a segmentation run logs a per-epoch grid search over the decision
threshold, MLflow records three additional metric series alongside the standard
``val_dice``:

- ``val_best_dice_at_threshold`` — Dice at the per-epoch argmax threshold τ*
  (in-sample on the validation set),
- ``val_optimal_threshold`` — the selected threshold τ* itself,
- ``val_dice_threshold_gain`` — ``val_best_dice_at_threshold − val_dice`` per epoch.

This module reduces those raw series to the two artefacts required by E4:

1. :func:`load_threshold_sweep` — one row per seed, collapsed at the **best swept
   epoch** (the epoch maximising ``val_best_dice_at_threshold``, which is also
   the checkpoint-selection criterion), together with the epoch-to-epoch
   variability of τ*.
2. :func:`paired_gain_stats` / :func:`threshold_stability` — paired in-sample
   gain statistics and threshold-stability diagnostics that determine whether a
   swept threshold is suitable for deployment.

All results are **validation-set, in-sample** (τ* is selected on the same 150
images against which it is scored); reported values represent an optimistic upper
bound and do not constitute a deployment claim. See the E4 notebook §7 for the
held-out test protocol.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from typing import cast
from numpy.typing import NDArray

from SkiNet.Utils.analysis.io import load_mlflow_tables
from SkiNet.Utils.analysis.stats import bootstrap_paired_ci

# MLflow metric keys consumed by this module.
SWEPT_KEY = "val_best_dice_at_threshold"
DICE_05_KEY = "val_dice"
TAU_KEY = "val_optimal_threshold"
GAIN_KEY = "val_dice_threshold_gain"
IOU_KEY = "val_iou"
TRAIN_DICE_KEY = "train_dice"
SPS_KEY = "perf/samples_per_sec"

# Default number of leading epochs excluded before measuring τ* variability.
# Early epochs are omitted because the threshold distribution is degenerate
# prior to sufficient model convergence and produces unreliable ROC statistics.
DEFAULT_WARMUP = 50


def _load_long(db_path: Path) -> pd.DataFrame:
    """Load metrics + run metadata from one MLflow SQLite store as a long DataFrame.

    :param db_path: Path to an MLflow ``.db`` tracking store.
    :return: Long DataFrame with columns ``run_uuid``, ``run_name``, ``status``,
             ``seed``, ``key``, ``value``, ``step`` — one row per metric point.
    :raises ValueError: If the store has no active runs or no metrics.
    """
    tables = load_mlflow_tables(db_path)
    runs = tables["runs"][["run_uuid", "run_name", "status"]].copy()
    runs["seed"] = runs["run_name"].str.extract(r"seed(\d+)")[0].astype(int)
    metrics = tables["metrics"][["run_uuid", "key", "value", "step"]]
    return metrics.merge(runs, on="run_uuid", how="left")


def _series_from_groups(
    groups: dict[tuple[str, str], pd.DataFrame], run_uuid: str, key: str
) -> NDArray[np.float64]:
    grp = groups.get((run_uuid, key))
    if grp is None:
        return np.array([], dtype=np.float64)
    return cast(NDArray[np.float64], grp.sort_values("step")["value"].to_numpy(dtype=float))


def load_threshold_sweep(
    *dbs: Path,
    part_split: int | None = None,
    warmup: int = DEFAULT_WARMUP,
) -> pd.DataFrame:
    """Construct the per-seed threshold-sweep table — one row per run at the best swept epoch.

    For every run, the **best swept epoch** is ``argmax(val_best_dice_at_threshold)``
    (the checkpoint-selection criterion). At that epoch the table records the swept
    Dice, the fixed-0.5 Dice for the identical model and weights, their in-sample
    gain, and the selected threshold τ*. It additionally computes the epoch-to-epoch
    variability of τ* over the post-warm-up tail, which :func:`threshold_stability`
    uses as the primary deployability diagnostic.

    Multiple databases are concatenated, so a multi-part experiment (e.g. E4 part 1
    and part 2) can be loaded in a single call. Pass ``part_split`` to label each
    seed as ``"P1"`` (``seed <= part_split``) or ``"P2"`` for per-part reporting.

    :param dbs: One or more MLflow ``.db`` paths.
    :param part_split: Seed boundary for the ``part`` label; if ``None``, every
                       row is tagged ``"P1"``.
    :param warmup: Number of leading epochs excluded before measuring τ* variability
                   (population SD, ``ddof=0``). Defaults to :data:`DEFAULT_WARMUP`.
    :return: DataFrame with one row per seed, columns ``seed``, ``part``,
             ``best_ep`` (1-based), ``val_best_dice_at_threshold``, ``val_dice``,
             ``val_dice_gain``, ``val_optimal_threshold``, ``val_tau_wander_sd``,
             ``val_tau_min``, ``val_tau_max``,
             ``val_iou``, ``gen_gap``, ``samples_per_sec``, ``n_epochs``;
             sorted by seed ascending. ``val_iou`` and ``gen_gap`` are read at
             the same best swept epoch; ``samples_per_sec`` is the run mean.
    """
    long_df = pd.concat([_load_long(Path(db)) for db in dbs], ignore_index=True)
    runs_meta = (
        long_df[["run_uuid", "seed", "status"]]
        .drop_duplicates()
        .sort_values("seed")
        .reset_index(drop=True)
    )
    groups = {k: v for k, v in long_df.groupby(["run_uuid", "key"])}

    rows = []
    for _, x in runs_meta.iterrows():
        swept = _series_from_groups(groups, x.run_uuid, SWEPT_KEY)
        d05 = _series_from_groups(groups, x.run_uuid, DICE_05_KEY)
        tau = _series_from_groups(groups, x.run_uuid, TAU_KEY)
        iou = _series_from_groups(groups, x.run_uuid, IOU_KEY)
        train_dice = _series_from_groups(groups, x.run_uuid, TRAIN_DICE_KEY)
        sps = _series_from_groups(groups, x.run_uuid, SPS_KEY)
        bi = int(np.argmax(swept))
        tail = tau[warmup:]
        rows.append(
            dict(
                seed=int(x.seed),
                part="P1" if part_split is None or x.seed <= part_split else "P2",
                best_ep=bi + 1,
                val_best_dice_at_threshold=float(swept[bi]),
                val_dice=float(d05[bi]),
                val_dice_gain=float(swept[bi] - d05[bi]),
                val_optimal_threshold=float(tau[bi]),
                val_tau_wander_sd=float(tail.std()) if tail.size else float("nan"),
                val_tau_min=float(tail.min()) if tail.size else float("nan"),
                val_tau_max=float(tail.max()) if tail.size else float("nan"),
                # Companion sanity metrics, all read at the same best swept epoch.
                val_iou=float(iou[bi]) if iou.size > bi else float("nan"),
                gen_gap=float(train_dice[bi] - d05[bi]) if train_dice.size > bi else float("nan"),
                samples_per_sec=float(sps.mean()) if sps.size else float("nan"),
                n_epochs=len(swept),
            )
        )
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def epoch_trajectories(*dbs: Path, keys: tuple[str, ...], part_split: int | None = None) -> dict[int, dict]:
    """Return per-seed, per-epoch metric trajectories for use in training-curve figures.

    Complements :func:`load_threshold_sweep`, which collapses each run to a single
    best-epoch row; this function retains the full epoch history so that figures
    can illustrate how τ* and the Dice gain evolve over the course of training.

    :param dbs: One or more MLflow ``.db`` paths.
    :param keys: Metric keys to extract per seed (e.g. ``(TAU_KEY, GAIN_KEY)``).
    :param part_split: Seed boundary for the ``part`` label; ``None`` → all ``"P1"``.
    :return: Dict mapping ``seed`` → ``{"part": str, <key>: np.ndarray, ...}``,
             ordered by seed ascending.
    """
    long_df = pd.concat([_load_long(Path(db)) for db in dbs], ignore_index=True)
    runs_meta = (
        long_df[["run_uuid", "seed"]].drop_duplicates().sort_values("seed")
    )
    groups = {k: v for k, v in long_df.groupby(["run_uuid", "key"])}
    out: dict[int, dict] = {}
    for _, x in runs_meta.iterrows():
        seed = int(x.seed)
        entry: dict = {"part": "P1" if part_split is None or seed <= part_split else "P2"}
        for key in keys:
            entry[key] = _series_from_groups(groups, x.run_uuid, key)
        out[seed] = entry
    return out


def paired_gain_stats(
    per_seed: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    random_state: int | np.random.Generator = 42,
) -> dict[str, float | int | list]:
    """Paired in-sample gain statistics (swept − 0.5) computed from the per-seed table.

    Because the swept and fixed-0.5 scores derive from the same model weights at
    the same epoch, the only source of variation between the two conditions is τ,
    yielding a clean paired comparison. Reports the mean gain, a BCa bootstrap
    95 % CI on the mean (consistent with the E2 protocol), the Wilcoxon signed-rank
    p-value, Cohen's d_z, and the count of seeds with a positive gain. A significant
    result is necessary but **not** sufficient evidence that threshold calibration
    improves performance on unseen data — the threshold is selected on the same
    images against which it is scored.

    :param per_seed: Output of :func:`load_threshold_sweep`.
    :param n_resamples: Bootstrap resamples for the BCa CI. Default 10 000.
    :param random_state: Seed or Generator for reproducibility.
    :return: Dict with ``n``, ``per_seed_val_dice_gain`` (list), ``mean``, ``ci_lo``,
             ``ci_hi``, ``wilcoxon_p``, ``cohen_dz``, ``n_positive``.
    """
    g = per_seed["val_dice_gain"].to_numpy()
    n = g.size
    sd = g.std(ddof=1)
    ci_lo, ci_hi = bootstrap_paired_ci(g, n_resamples=n_resamples, random_state=random_state)
    return {
        "n": n,
        "per_seed_val_dice_gain": np.round(g, 4).tolist(),
        "mean": float(g.mean()),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "wilcoxon_p": float(stats.wilcoxon(per_seed["val_best_dice_at_threshold"], per_seed["val_dice"]).pvalue),
        "cohen_dz": float(g.mean() / sd) if sd > 0 else float("nan"),
        "n_positive": int((g > 0).sum()),
    }


def threshold_stability(per_seed: pd.DataFrame) -> dict[str, float]:
    """Threshold-stability diagnostics — the operative deployability signal.

    A calibration gain is suitable for deployment only if the threshold that
    produced it is stable. Two independent failure modes are assessed:

    - **Across seeds** — τ* values dispersed around 0.5 with no consistent
      directional offset indicate the absence of a systematic miscalibration
      that a single tuned threshold could correct. Reported as ``tau_median``,
      ``tau_mean``, ``tau_sd``, ``tau_dist_from_half``.
    - **Within a seed, across epochs** — τ* that varies substantially between
      adjacent epochs is co-varying with validation sampling noise rather than
      reflecting a stable property of the decision boundary. Reported as
      ``mean_wander_sd`` and the global range ``[tau_global_min, tau_global_max]``.

    :param per_seed: Output of :func:`load_threshold_sweep`.
    :return: Dict of the stability summary statistics described above.
    """
    tau = per_seed["val_optimal_threshold"].to_numpy()

    stab = {
        "tau_median": float(np.median(tau)),
        "tau_mean": float(tau.mean()),
        "tau_sd": float(tau.std(ddof=1)),
        "tau_dist_from_half": float(abs(np.median(tau) - 0.5)),
        "mean_wander_sd": float(per_seed["val_tau_wander_sd"].mean()),
        "tau_global_min": float(per_seed["val_tau_min"].min()),
        "tau_global_max": float(per_seed["val_tau_max"].max()),
    }

    print('Across seeds (does τ* agree run-to-run?)')
    print(f"  median τ*           : {stab['tau_median']:.3f}")
    print(f"  mean τ*             : {stab['tau_mean']:.3f}")
    print(f"  SD of τ* (seeds)    : {stab['tau_sd']:.3f}")
    print(f"  |median − 0.5|      : {stab['tau_dist_from_half']:.3f}")
    print()
    print('Within a run, across epochs (does τ* converge?)')
    print(f"  mean within-run SD  : {stab['mean_wander_sd']:.3f}")
    print(f"  global τ* range     : [{stab['tau_global_min']:.2f}, {stab['tau_global_max']:.2f}]")
    print()
    near_half = stab['tau_dist_from_half'] < stab['tau_sd']          # median indistinguishable from 0.5
    wanders = stab['mean_wander_sd'] > 0.05                        # τ* not converged within a run
    print('Verdict on stability:')
    print(f"  median τ* within 1 SD of 0.5 ? {near_half}  → "
          f"{'no systematic miscalibration' if near_half else 'consistent offset from 0.5'}")
    print(f"  τ* wanders within a run ?      {wanders}  → "
          f"{'τ* tracks val noise, not converged' if wanders else 'τ* converged'}")

    return stab
