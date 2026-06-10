"""Canonical column names for the long-format runs DataFrame.

These are the structural keys of the table produced by
:func:`SkiNet.Utils.analysis.aggregation.load_runs` and consumed across the
analysis package (``stats``, ``plotting``, ``reporting``).

The metric columns (``val_dice_max``, ``val_dice_tail_mean``, …) are generated dynamically from the ``monitor``
"""

from __future__ import annotations

SEED = "seed"
ARCH = "arch"

# Well-known metric columns. These are generated dynamically from the
# ``monitor`` passed to :func:`load_runs` (so they are not *structural* keys
# like SEED/ARCH), but the names are stable across experiments and referenced
# by many notebooks. Defining them once keeps a metric rename to a single edit.
VAL_DICE_MAX = "val_dice_max"
VAL_DICE_TAIL_MEAN = "val_dice_tail_mean"
VAL_DICE_TAIL_STD = "val_dice_tail_std"
VAL_IOU_MAX = "val_iou_max"
GENERALIZATION_GAP_FINAL = "generalization_gap_final"
SAMPLES_PER_SEC = "samples_per_sec"
DURATION_MIN = "duration_min"

# ── Batch size sweep (E0) ─────────────────────────────────────────────────────

EXPECTED_BATCH_SIZES: list[int] = [4, 8, 16, 32, 64, 128]
MAX_EPOCHS_SWEEP: int = 10
N_TRAIN_IMAGES_ISIC2017: int = 2000
EFFICIENCY_THRESHOLD_PCT: float = 80.0
REFERENCE_BS: int = 4

STEP_METRICS: list[str] = [
    "perf/samples_per_sec",
    "perf/time_per_step_ms",
    "system/gpu_mem_allocated_gb",
    "system/gpu_util_percent",
    "train_loss_step",
    "epoch",
]

TIDY_COLS: list[str] = [
    "experiment", "batch_size", "run_uuid", "step",
    "samples_per_sec", "time_per_step_ms",
    "epoch_idx", "gpu_mem_gb", "gpu_util_pct", "train_loss",
    "is_outlier", "outlier_reason",
]
