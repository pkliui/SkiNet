"""Held-out test-set scoring for a trained segmentation checkpoint.

This is the **test-only** half of the E4/EF split: the decision threshold is found
on the validation set elsewhere (the E4 threshold-sweep notebook), and this module
just *scores* an already-selected checkpoint on the disjoint test split at whatever
threshold(s) it is handed. Nothing here searches for a threshold.

All Dice/IoU are **per-image mean** (ISIC-2017 official averaging): the metric is
computed per image and then averaged over images, never pooled across pixels.

Used by both the ``EF`` notebook (single selected checkpoint, inline inference) and
``calibrate_threshold.py`` (the same scoring over a glob of checkpoints).
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from glob import glob
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


# ----------------------------- metrics -------------------------------------- #
def per_image_dice_iou(probs: torch.Tensor, masks: torch.Tensor, thr: float,
                       eps: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """Per-image Dice and IoU at a fixed threshold (ISIC-official averaging).

    :param probs: Predicted foreground probabilities, shape ``[N, P]`` (N images,
        P pixels each).
    :param masks: Binary ground-truth masks, same shape.
    :param thr: Decision threshold applied to ``probs``.
    :param eps: Smoothing constant guarding the empty-mask / empty-prediction case.
    :return: ``(dice, iou)`` as length-``N`` numpy arrays — one score per image, not
        pooled. Take ``.mean()`` for the ISIC per-image mean.
    """
    preds = (probs >= thr).float()
    inter = (preds * masks).sum(dim=1)
    psum, msum = preds.sum(dim=1), masks.sum(dim=1)
    dice = (2 * inter + eps) / (psum + msum + eps)
    iou = (inter + eps) / (psum + msum - inter + eps)
    return dice.cpu().numpy(), iou.cpu().numpy()


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap 95 % CI on the mean of ``values``.

    :param values: Per-image scores to resample (e.g. the per-image Dice array).
    :param n_boot: Number of bootstrap resamples.
    :param seed: Seed for the resampling RNG (reproducible CIs).
    :return: ``(lo, hi)`` 2.5 / 97.5 percentile bounds on the resampled mean.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score_at_thresholds(
    probs: torch.Tensor,
    masks: torch.Tensor,
    thresholds: Iterable[float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Score one checkpoint's test predictions at each given threshold.

    Pure scoring — the thresholds are inputs (found on val upstream), never searched
    here. For each threshold it reports the per-image mean Dice/IoU and a bootstrap
    95 % CI on the mean Dice.

    :param probs: Per-image test probabilities ``[N, P]`` (e.g. from
        :func:`collect_probs`).
    :param masks: Matching binary masks ``[N, P]``.
    :param thresholds: Decision thresholds to evaluate (e.g. ``[0.5, tau_val]``).
    :param n_boot: Bootstrap resamples for the CI.
    :param seed: Bootstrap RNG seed.
    :return: One row per threshold with columns ``threshold``, ``dice``, ``iou``,
        ``dice_lo``, ``dice_hi`` (Dice/IoU are per-image means).
    """
    rows = []
    for thr in thresholds:
        d, i = per_image_dice_iou(probs, masks, float(thr))
        lo, hi = bootstrap_ci(d, n_boot, max(seed, 0))
        rows.append({
            "threshold": float(thr),
            "dice": float(d.mean()),
            "iou": float(i.mean()),
            "dice_lo": lo,
            "dice_hi": hi,
        })
    return pd.DataFrame(rows)


# ----------------------------- model / inference ----------------------------- #
def load_uncompiled(cfg: Any, ckpt: str | Path) -> torch.nn.Module:
    """Build the model **uncompiled** and load weights from a checkpoint.

    Forces ``use_torch_compile = False`` (so no torch.compile / inductor / nvcc is
    invoked) and strips the ``_orig_mod.`` key prefix that ``torch.compile`` adds, so
    checkpoints saved from a compiled model load cleanly.

    :param cfg: Experiment config (mutated in place to disable compilation).
    :param ckpt: Path to the ``.ckpt`` file.
    :return: The eval-ready :class:`torch.nn.Module` with weights loaded ``strict``.
    """
    from SkiNet.ML.model.lightning_model import build_lightning_model  # lazy: heavy import

    cfg.trainconfig.use_torch_compile = False
    model = build_lightning_model(cfg)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model


@torch.no_grad()
def collect_probs(model: torch.nn.Module, loader: Any, device: torch.device
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inference once over a loader, returning per-image probabilities + masks.

    :param model: Model to evaluate (set to ``eval`` and moved to ``device``).
    :param loader: Dataloader yielding ``{"image", "mask"}`` batches.
    :param device: Inference device.
    :return: ``(probs, masks)`` each ``[N, P]`` on CPU — sigmoid probabilities and
        binarised masks, flattened per image.
    """
    model.eval().to(device)
    ps, ms = [], []
    for batch in loader:
        x = batch["image"].to(device)
        m = batch["mask"].to(device)
        probs = torch.sigmoid(model(x))
        n = probs.shape[0]
        ps.append(probs.reshape(n, -1).cpu())
        ms.append((m.reshape(n, -1) >= 0.5).float().cpu())
    return torch.cat(ps), torch.cat(ms)


# ----------------------------- checkpoint discovery -------------------------- #
def build_ckpt_map(
    *dbs: Path | str,
    glob_pattern: str,
    project_root: Path | str,
) -> dict[int, Path]:
    """Map each seed to its best-checkpoint file path.

    The seed lives in the MLflow run name (``...seed107...``) but not in the
    checkpoint path, which is keyed by run UUID. This reads ``run_uuid → seed`` from
    the tracking stores, globs the checkpoint files, and joins the two on the 32-hex
    UUID embedded in each checkpoint path.

    :param dbs: One or more MLflow ``.db`` paths (run-name → seed source).
    :param glob_pattern: Absolute glob for the best-checkpoint ``.ckpt`` files.
    :param project_root: Repo root, used only to keep returned paths absolute.
    :return: Dict mapping ``seed`` → checkpoint :class:`~pathlib.Path`, for every
        checkpoint whose UUID resolves to a seed.
    """
    project_root = Path(project_root)
    uuid2seed: dict[str, int] = {}
    for db in dbs:
        with closing(sqlite3.connect(str(db))) as con:
            for uuid, name in con.execute("SELECT run_uuid, name FROM runs").fetchall():
                m = re.search(r"seed(\d+)", name or "")
                if m:
                    uuid2seed[uuid] = int(m.group(1))

    ckpt_map: dict[int, Path] = {}
    for f in sorted(glob(glob_pattern)):
        m = re.search(r"/([0-9a-f]{32})/", f)
        if m and m.group(1) in uuid2seed:
            ckpt_map[uuid2seed[m.group(1)]] = Path(f)
    return ckpt_map
