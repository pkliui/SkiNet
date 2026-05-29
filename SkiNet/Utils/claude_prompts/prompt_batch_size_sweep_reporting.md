You are an expert ML engineer writing a professional Jupyter notebook to analyze a batch size sweep experiment for a medical image segmentation model.

- If any required data, file, or metric is unavailable, explicitly say so and use a placeholder with a TODO instead of guessing. Do not infer, assume, or fabricate values, paths, or metrics. Only use explicitly defined information.

- Before writing the notebook, make a brief implementation checklist. Then generate the notebook in executable order. Ensure every downstream cell can run even when real MLflow data is unavailable by using the placeholder DataFrame schema exactly as specified.

---

## 1. Project Context

A 2D U-Net (UNet2D) was trained on the **ISIC 2017** dermoscopic segmentation dataset
(~2000 training images, 256×256 px) to sweep over batch sizes and determine the optimal
batch size for a **dual T4 GPU** setup. Two independent sweeps were conducted:

| Experiment | Description | DB file |
|---|---|---|
| No-augmentation | Baseline sweep, training-time augmentations **off** | `mlflow-the-batch-size-sweep-experiment.db` |
| With-augmentation | Same sweep, training-time augmentations **on** | `mlflow-the-batch-size-sweep-experiment-augm.db` |

Both sweeps share the same fixed configuration:

| Parameter | Value |
|---|---|
| Dataset | ISIC 2017, ~2000 training images, 256×256 px |
| Batch sizes tested | 4, 8, 16, 32, 64, 128 |
| Optimizer | AdamW, wd=1e-4; lr **scaled linearly with batch size** (see note below) |
| Max epochs | 10 (fixed epoch budget for throughput measurement) |
| LR scheduler | ReduceLROnPlateau on `val_best_dice_at_threshold`, patience=3, factor=0.5 — **disabled** (`use_lr_scheduler: False`) |
| GPU | 1× T4 (16 GB) per run (`fit/devices: 1`); target hardware for production is 2× T4 |
| DataLoader | num_workers=2, prefetch_factor=4, pin_memory=True |
| Architecture | UNet2D with local-refinement encoder and merge blocks (fixed throughout sweep) |
| Seed | Fixed (same for all runs) |

> **Note on learning rate — linear scaling rule is applied per batch size:**
> The code in `SkiNet/Utils/mlops/optuna_utils.py` (`scale_lr()`) applies the **linear
> scaling rule** anchored to the smallest batch size in the sweep (bs=4, lr=3e-4):
>
> | Batch size | `lr_optuna_sampled` | `lr_actually_used` | Scale factor |
> |---|---|---|---|
> | 4 | 3e-4 | 3e-4 | ×1 |
> | 8 | 3e-4 | 6e-4 | ×2 |
> | 16 | 3e-4 | 1.2e-3 | ×4 |
> | 32 | 3e-4 | 2.4e-3 | ×8 |
> | 64 | 3e-4 | 4.8e-3 | ×16 |
> | 128 | 3e-4 | 9.6e-3 | ×32 |
>
> **Impact on throughput measurement (`perf/samples_per_sec`):** None. The LR value
> controls gradient magnitude, not compute scheduling. `samples_per_sec` is determined
> purely by batch size, DataLoader speed, and GPU compute — it is independent of LR
> unless the training diverges (NaN loss). The throughput comparison across batch sizes
> is therefore valid.
>
> **Impact on loss convergence (Section 5):** The scaled LR may be too aggressive at
> large batch sizes (e.g., 9.6e-3 at bs=128), potentially causing unstable loss curves.
> Section 5 must flag any such instability explicitly. The LR will be re-tuned from
> scratch in the subsequent Optuna HPO sweep once the batch size is fixed. IN other words, verify that the runs at large batch sizes didn't exhibit loss instability — because if they did, the scaling breakdown you observe might be LR-induced rather than a genuine hardware saturation effect.
>
> **Note on target hardware — DDP:** The production target is **Distributed Data Parallel (DDP)**
> training across 2× T4 GPUs. In DDP, each GPU processes its own mini-batch independently,
> computes gradients, and gradients are synchronised (averaged) across GPUs before the weight
> update. The effective batch size in DDP is `batch_size × n_gpus`. These sweep runs were
> conducted on a **single GPU** to measure raw per-GPU throughput; the DDP scaling implication
> (that effective batch size doubles) is discussed in the recommendation section.

> **Goal of this analysis:** Following the Google Deep Learning Tuning Playbook guidance on
> batch size selection, identify the optimal batch size as the **largest batch size that
> still produces a proportional increase in training throughput** (i.e., the one that lies
> at or just before the "critical batch size" where perfect scaling breaks down). Batch size
> should **not** be treated as a hyperparameter for tuning validation performance — the
> focus is entirely on training speed and GPU utilisation.

---

## 2. Data Sources & Schema

### 2.1 MLflow database layout

Both databases share the same SQLite schema. Relevant tables:

| Table | Key columns |
|---|---|
| `runs` | `run_uuid`, `name`, `status`, `start_time`, `end_time`, `lifecycle_stage` |
| `params` | `key`, `value`, `run_uuid` |
| `metrics` | `key`, `value`, `step`, `timestamp`, `run_uuid` |

Connect with:
```python
import sqlite3
con = sqlite3.connect("<path_to_db>")
```

**Load only `lifecycle_stage = 'active'` runs.**

### 2.2 Param keys (exact strings)

| Param key | Description |
|---|---|
| `batch_size` | Batch size for this run (int as string: "4", "8", …, "128") |
| `fit/batch_size` | Same value, logged by Lightning trainer |
| `fit/max_epochs` | Number of epochs (10 for all runs) |
| `fit/num_workers` | DataLoader workers |
| `lr` | Effective learning rate used |
| `prefetch_factor` | DataLoader prefetch factor |

Use `batch_size` as the primary grouping key.

### 2.3 Metric keys (exact strings)

**Step-level metrics** — one record per training step, stored in `metrics` table:

| Metric key | Description |
|---|---|
| `perf/samples_per_sec` | Training throughput in images/sec at each step |
| `perf/time_per_step_ms` | Wall-clock time per step in milliseconds |
| `train_loss_step` | Per-step training loss |
| `train_dice` | Per-step training Dice |
| `train_iou` | Per-step training IoU |
| `train_loss_epoch` | Per-epoch training loss (logged once per epoch) |
| `epoch` | Current epoch index (0-based) |
| `system/gpu_mem_allocated_gb` | GPU memory actually allocated (GB) |
| `system/gpu_mem_reserved_gb` | GPU memory reserved by CUDA allocator (GB) |
| `system/gpu_util_percent` | GPU utilisation (%) |

**Summary metrics** — logged once at run end (last-step or best-step snapshot):

| Metric key | Description |
|---|---|
| `final/perf/samples_per_sec` | Throughput at the very last step |
| `final/perf/time_per_step_ms` | Time per step at the very last step |
| `best_perf/samples_per_sec` | Highest throughput seen across all steps |
| `final/train_dice` | Train Dice at the very last step |
| `final/train_loss` | Train loss at the very last step |

> ⚠️ **Do not rely on `final/` or `best_perf/` metrics as the primary throughput estimate.**
> The `final/` values reflect a single step that may be an epoch-boundary outlier (see
> Section 3). Always compute throughput estimates from the cleaned step-level time series.

---

## 3. ⚠️ Critical Known Issue — Outliers at Step Boundaries

Three distinct categories of outliers corrupt `perf/samples_per_sec` and
`perf/time_per_step_ms`. All must be identified and **marked** (never silently dropped)
before computing any throughput statistics.

### 3.1 Step-0: two-entry artefact (compilation + summary metric)

**Step 0 in every run has exactly two DB entries** in the `metrics` table with the same
`step=0` but different `timestamp` values:

1. **Early timestamp** — the actual first training step, which is catastrophically slow
   due to CUDA kernel compilation and DataLoader worker spin-up.
   In `perf/samples_per_sec`: values of 0.3–100 samples/sec (vs. 150–200 steady state).
   In `perf/time_per_step_ms`: values of 224–30 562 ms (vs. 30–900 ms steady state).

2. **Late timestamp** — a run-end summary metric logged at step=0 by the training script
   (identical in value to `final/perf/samples_per_sec`).
   In `perf/samples_per_sec`: 121–657 samples/sec (the final-step throughput).

Observed values per batch size:

| Batch size | Step-0 early (samples/sec) | Step-0 late (samples/sec) | Steady-state mean (samples/sec) |
|---|---|---|---|
| 4 | ~17.8 | ~121.6 | ~124 |
| 8 | ~0.3 | ~224.1 | ~201 |
| 16 | ~0.6 | ~165.1 | ~167 |
| 32 | ~61.7 | ~312.8 | ~160 |
| 64 | ~78.2 | ~657.3 | ~175 |
| 128 | ~99.6 | ~233.4 | ~153 |

**Rule:** Exclude ALL entries where `step == 0`, regardless of timestamp.
Do not attempt to keep the late-timestamp entry — its value is already represented
in the final-step of the training sequence and would duplicate data.

### 3.2 Epoch-boundary pair: last-of-epoch HIGH spike + first-of-next LOW dip

At every epoch boundary, two adjacent steps are anomalous in opposite directions:

- **Last step of each epoch** (`step = n × spe − 1`, n = 1…9) — records an
  **artificially HIGH** `samples_per_sec`. The likely cause is that the timing
  measurement for this step captures a shorter-than-usual compute window because
  the epoch-end validation loop begins immediately after. The step appears "faster"
  than it really is.

- **First step of the next epoch** (`step = n × spe`, n = 1…9) — records an
  **artificially LOW** `samples_per_sec` (HIGH `time_per_step_ms`). This step
  absorbs the overhead of DataLoader reset, full validation pass, checkpoint save,
  and LR scheduler evaluation.

Steps per epoch by batch size (derived from ~2000 training images / batch size):

| Batch size | Steps/epoch (spe) |
|---|---|
| 4 | 500 |
| 8 | 250 |
| 16 | 125 |
| 32 | 63 |
| 64 | 32 |
| 128 | 16 |

Observed examples (samples/sec):

| Batch size | step n×spe−1 (HIGH) | step n×spe (LOW) | Steady-state mean |
|---|---|---|---|
| 8 | ~228 | ~131 | ~201 |
| 32 | ~330 | ~156 | ~160 |
| 64 | ~677 | ~158 | ~175 |

Note for bs=8: the HIGH outlier (~228 samples/sec) is the value that visually
exceeds the clean steady-state band by the most. All values above approximately
220 samples/sec in the bs=8 run are epoch-last-step artefacts (or the step-0
late-timestamp entry).

**Rule:** For each run, compute `spe = total_steps / max_epochs` (integer).
Mark as outliers: all `step == n * spe - 1` and `step == n * spe` for n = 1, 2, …, 9.

### 3.3 Outlier detection implementation

```python
def get_outlier_steps(spe: int, max_epochs: int) -> set:
    """Return step indices to exclude: step 0, epoch-last, and epoch-first steps."""
    outlier_steps = {0}
    for n in range(1, max_epochs + 1):
        outlier_steps.add(n * spe - 1)   # last step of epoch n (HIGH artefact)
        outlier_steps.add(n * spe)        # first step of epoch n+1 (LOW artefact)
    return outlier_steps
```

After applying the rule-based exclusions above, apply one additional pass: an
**IQR filter** (exclude values below Q1 − 3×IQR or above Q3 + 3×IQR) on the
remaining `perf/samples_per_sec` values within each run as a catch-all.
Log the count of points removed at each stage in a Markdown cell.
**Never silently drop points — mark them with `is_outlier=True` and keep in the DataFrame.**

Use the **median** of the cleaned (non-outlier) distribution as the throughput
estimate for each batch size. Report the 10th–90th percentile range as a stability
indicator.

---

## 4. Notebook Structure

### Section 0 — Design of Experiment & Motivation *(Markdown only — opens the notebook)*

This section must appear **before any code**, as the first cell of the notebook.
Explain in 3–5 sentences each:

- **Why batch size governs training speed, not model quality** (Shallue et al. 2018 /
  Tuning Playbook): with well-tuned hyperparameters and sufficient steps, any batch size
  can reach the same final performance. Batch size selection is therefore a pure throughput
  optimisation problem.
- **What "perfect scaling" means:** doubling the batch size should double throughput
  (halve time per step) as long as the GPU is not saturated. This is the linear scaling
  regime.
- **Why there is a critical batch size** beyond which scaling breaks down: I/O bottlenecks,
  memory bandwidth saturation, or inter-GPU synchronisation overhead. The optimal batch
  size is the largest that still gives a meaningful throughput gain.
- **Why augmentations are tested separately:** augmentations add CPU pre-processing
  overhead and may shift the bottleneck from GPU to CPU/DataLoader, potentially changing
  which batch size saturates the GPU first.
- **Why LR was scaled linearly with batch size in these sweep runs, and why it does not
  affect the conclusion:** The Tuning Playbook states that optimizer hyperparameters
  (especially LR) interact most strongly with batch size and must be re-tuned separately
  for each batch size — the linear scaling rule (`lr ∝ bs`) is only a heuristic
  approximation. Crucially, `perf/samples_per_sec` (the selection criterion) depends
  purely on compute and I/O scheduling; it is independent of LR value provided training
  does not diverge. The throughput comparison is therefore valid regardless of how LR was
  set. The loss convergence check (Section 5) exists precisely to verify that no run
  diverged due to an over-scaled LR. Once the optimal batch size is chosen from this
  sweep, **all optimizer hyperparameters must be re-tuned from scratch** via the
  subsequent Optuna HPO study — the scaled LRs used here should not be carried forward.
- **Why these sweep runs were conducted on a single GPU:** the per-GPU throughput is
  measured here. When deployed under DDP across 2× T4, the effective batch size doubles
  (`bs × 2`), which must be considered when interpreting the recommendation.

---

### Section 1 — Setup & Data Loading

- Import all dependencies: `sqlite3`, `pandas`, `numpy`, `matplotlib`, `seaborn`,
  `scipy.stats`
- Define DB paths as constants at the top; load both DBs
- For each DB, fetch all active runs and join with `batch_size` param
- For each run, load the full step-level time series for:
  `perf/samples_per_sec`, `perf/time_per_step_ms`,
  `system/gpu_mem_allocated_gb`, `system/gpu_util_percent`
- Produce one tidy long-format DataFrame per DB with columns:
  `batch_size` (int), `step`, `samples_per_sec`, `time_per_step_ms`,
   `epoch_idx`, `gpu_mem_gb`, `gpu_util_pct`, `is_outlier` (bool)
- Add a boolean `is_outlier` column using the rules from Section 3 of this prompt;
  never drop rows, only mark them. Downstream cells filter on `is_outlier == False`.

**MLflow fallback:** If either DB cannot be read, scaffold with `# TODO:` comments and
a placeholder DataFrame (columns as above, 6 batch sizes × 500 dummy steps each) so all
downstream cells execute end-to-end.

---

### Section 2 — Outlier Audit *(Markdown + code)*

- Plot `perf/samples_per_sec` vs `step` for each batch size on a 2×3 subplot grid,
  with outlier points highlighted in red and clean points in blue.
  This makes the three outlier categories (step-0 pair, epoch-last HIGH, epoch-first LOW)
  visually distinct for each batch size.
- Plot `time_per_step_ms` vs `step` on the same grid layout for cross-verification.
- Add a Markdown summary table showing, per batch size: total DB entries at step=0,
  rule-based outlier count, IQR outlier count, total outlier percentage, and
  median clean throughput.
- This section is diagnostic only — do not draw conclusions here.

---

### Section 3 — Throughput Analysis *(central result)*

All statistics in this section use only **clean (non-outlier) steps**.

#### 3.1 Median throughput vs batch size

- Bar chart: median `samples_per_sec` (y-axis) vs batch size (x-axis, log2 scale),
  error bars = [10th, 90th] percentile range.
- Overlay the **perfect-scaling reference line** anchored at bs=4:
  `perfect_scaling[bs] = median_throughput[4] × (bs / 4)`.
- Show both experiments (no-augmentation, with-augmentation) as grouped bars or
  side-by-side subplots.

#### 3.2 Scaling efficiency

Define scaling efficiency as:
```
efficiency[bs] = median_throughput[bs] / (median_throughput[reference_bs] × (bs / reference_bs))
```
where `reference_bs = 4`.

- Line chart: scaling efficiency (y-axis, %) vs batch size (x-axis, log2 scale).
  Efficiency = 100% means perfect linear scaling.
- Add a horizontal dashed line at 100% (perfect scaling) and at a practical threshold
  (e.g. 80%) below which scaling is considered poor.
- Mark the **critical batch size** — the last batch size before efficiency drops below
  the practical threshold — with a vertical dashed line.

#### 3.3 Time per step vs batch size

- Line chart: median `time_per_step_ms` vs batch size (log2 scale).
- If time per step is approximately constant as batch size doubles, the pipeline is
  compute-bound (good). If it grows, there is a bottleneck.

#### 3.4 Throughput time series (stability check)

- For each batch size, plot `samples_per_sec` vs step (clean steps only) with a
  horizontal median line and shaded [10th, 90th] percentile band.
- Purpose: verify that throughput is stable within a run (no drift), which validates
  using the median as a representative estimate.

---

### Section 4 — GPU Utilisation & Memory

- Bar chart: median `gpu_util_percent` per batch size (clean steps, both experiments).
- Bar chart: median `gpu_mem_allocated_gb` per batch size.
- Add a horizontal reference line at T4 memory limit (16 GB) and note headroom.
- A batch size is only feasible for DDP across 2× T4 if
  `peak_gpu_mem_allocated_gb ≤ 16 GB` per card (each GPU processes its own `batch_size`
  images, not `batch_size × 2`). Flag any batch size that approaches or exceeds 12 GB as
  a memory risk for DDP.

---

### Section 5 — Training Loss Convergence (Context Only)

> **Important framing:** Per the Tuning Playbook, batch size should **not** be used to
> tune validation performance. This section is included solely as a sanity check that all
> batch sizes achieve reasonable convergence within 10 epochs — not to rank batch sizes by
> loss.

- Small-multiples plot (2×3 grid): `train_loss_step` vs step for each batch size, with
  outlier steps greyed out.
- Overlay per-epoch mean loss as a bold line.
- A Markdown cell must explicitly state: "These loss curves are presented for sanity
  checking only. Batch size selection is based exclusively on throughput criteria from
  Section 3."

---

### Section 6 — Analysis & Recommendation *(Markdown + supporting code)*

Address each point with 2–4 sentences, citing specific numbers from the cleaned data:

- **Throughput scaling verdict:** Does throughput scale linearly up to some batch size,
  then plateau or drop? State the exact batch sizes where scaling efficiency falls below
  80%. Is the pattern consistent across both experiments?

- **Critical batch size identification:** State the recommended batch size — the largest
  one that still shows good scaling efficiency in both experiments. If the two experiments
  disagree, discuss why (augmentation overhead changes the CPU/GPU balance) and recommend
  the more conservative value.

- **GPU utilisation assessment:** Are smaller batch sizes leaving the GPU under-utilised?
  Is the GPU already saturated at some batch size, making larger batches wasteful?

- **Memory headroom:** Is the recommended batch size safe for a 2× T4 DDP setup? Quote
  the peak memory per GPU and the remaining headroom.

- **Practical implication for the Optuna HPO sweep:** Per the Tuning Playbook, the
  optimal values of optimizer hyperparameters (LR, weight decay, momentum) are sensitive
  to batch size and must be re-tuned independently for the chosen batch size — the linear
  scaling heuristic applied in this sweep is not a substitute. State explicitly that the
  chosen batch size will be **fixed** for all subsequent Optuna trials, and that the LR
  search range in that study should be centred around the linear-scaled value as a warm
  start but must be allowed to deviate freely.

- **Augmentation effect:** Does the presence of augmentations materially change the
  optimal batch size, or is the recommendation the same for both conditions? Quantify
  the throughput penalty of augmentation at the recommended batch size.

---

### Section 7 — Conclusion *(Markdown only)*

≤150 words. State:
1. The recommended batch size and the justification (last batch size before scaling efficiency
   degrades, within GPU memory limits).
2. Whether the recommendation is consistent across both augmented and non-augmented conditions.
3. The immediate next step: fix this batch size and re-tune LR (and other optimizer HPs)
   in the subsequent Optuna HPO sweep.

---

## 5. Global Style Rules

- Every major section opens with a Markdown cell: `##` header + 2–4 sentences explaining
  what is being done and why.
- All figures: title, x-axis label, y-axis label, legend. Use `seaborn` and `matplotlib`.
- No scope creep — only the 6 batch sizes in the two experiments; do not introduce new
  designs, datasets, or metrics not listed above.
- Use functions for repeated logic; no copy-paste blocks.
- All throughput summaries show: median, 10th percentile, 90th percentile, and outlier count.

---

## 6. Output Priorities

1. Outlier audit (Section 2) — establish trust in the data before any analysis.
2. Scaling efficiency chart (Section 3.2) — the central result; identifies the critical
   batch size as defined by the Tuning Playbook.
3. Median throughput vs batch size with perfect-scaling reference (Section 3.1).
4. GPU utilisation and memory headroom (Section 4) — feasibility gate for DDP.
5. Recommendation (Section 6) — grounded in numbers from Sections 3–4.

---

## 7. Output & Saving

- Save the notebook as: `batch_size_sweep_analysis_unet2d_isic2017.ipynb`
- Save location: `repos/SkiNet/analysis_results` (project root, alongside `workflow_guide.md`)
- Use `nbformat` to construct and save the notebook programmatically
- Ensure the notebook is fully executable end-to-end after saving
