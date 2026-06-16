# Training

## Set up training

- All settings are expected to be specified in `main_config.yaml`.
- Training settings live under the `trainconfig` section. Defaults, validation, and auto-derived values are managed through
{py:class}`SkiNet.ML.configs.train_configs.train_config.TrainConfig`.
- `precision` is auto-detected from `accelerator`: GPU/CUDA/MPS → `"16-mixed"`, CPU → `"32-true"`. Override explicitly if needed.
- For an Optuna sweep on GPU keep `precision: "16-mixed"`; for CPU sweeps either omit `precision` (auto-detected) or set a CPU-supported value.

### Key TrainConfig fields

| Field | Default | Description |
|---|---|---|
| `batch_size` | `8` | Samples per training batch |
| `num_workers` | auto (CPU count / DDP devices) | DataLoader workers; auto-divided among DDP processes |
| `pin_memory` | `True` on GPU, `False` on CPU/MPS | Auto-set from accelerator |
| `prefetch_factor` | `None` | Batches pre-loaded per worker; ignored when `num_workers=0` |
| `cache_in_ram` | `True` | Cache dataset in RAM before training; set `False` for large datasets (e.g. ISIC full split) |
| `use_torch_compile` | `False` | Wrap model with `torch.compile` for faster inference; first forward pass incurs JIT compilation overhead |
| `loss_name` | `BCE_DICE` | `BCE`, `DICE`, or `BCE_DICE` (equal 0.5/0.5 weight) |
| `optimizer_name` | `"adamw"` | `"adam"` or `"adamw"` |
| `lr` | `1e-4` | Base learning rate (linearly scaled by batch size in Optuna sweeps) |
| `weight_decay` | `1e-4` | L2 regularisation strength |
| `optimal_threshold` | `None` | `None` = sweep 51 thresholds each validation epoch; or a fixed float in [0, 1] |
| `max_epochs` | `1` | Maximum training epochs |
| `accelerator` | `"auto"` | Resolves to `gpu` / `mps` / `cpu` |
| `precision` | auto | Derived from accelerator; override if needed |
| `deterministic` | `True` | See [Reproducibility](#reproducibility) |
| `seed` | `42` | Global RNG seed passed to `L.seed_everything` |
| `use_lr_scheduler` | `True` | Enable/disable the LR scheduler |
| `scheduler_type` | `"reduce_on_plateau"` | `"reduce_on_plateau"` or `"cosine_annealing"` |
| `cosine_annealing_config.T_max` | `None` (→ `max_epochs`) | Period of cosine annealing; auto-set to `max_epochs` when `None` |
| `cosine_annealing_config.eta_min` | `1e-6` | Minimum learning rate at the end of each cosine cycle |

## Lightning model

The core training logic lives in {py:class}`SkiNet.ML.model.lightning_model.LightningModel` (a `L.LightningModule` subclass).

### Metrics

| Metric logged | Description |
|---|---|
| `train_loss` / `val_loss` / `test_loss` | Epoch-mean loss for the configured `loss_name` |
| `train_dice` / `val_dice` / `test_dice` | `BinaryF1Score` at the fixed or swept threshold |
| `train_iou` / `val_iou` / `test_iou` | `BinaryJaccardIndex` |
| `val_best_dice_at_threshold` | Best dataset-aggregated Dice found across the threshold sweep |
| `val_mean_dice_per_image` | Mean of per-image Dice at the best threshold; **schema default monitor** (`MetricsKey.default_monitor()`) |
| `val_optimal_threshold` | Threshold that achieved `val_best_dice_at_threshold` |
| `val_dice_threshold_gain` | Dice gain from using the optimal threshold vs. 0.5 |
| `val_threshold_used` | Threshold actually applied during validation |

### Optimizer and scheduler

- **Optimizer:** Adam or AdamW (set via `optimizer_name`), with configurable `lr` and `weight_decay`.
- **Scheduler:** controlled by `scheduler_type` (`"reduce_on_plateau"` or `"cosine_annealing"`); toggled via `use_lr_scheduler`.
  - `"reduce_on_plateau"` (schema default): `ReduceLROnPlateau`, mode `"max"`, schema defaults patience `5` / factor `0.5` (the shipped `main_config.yaml` sets patience `3`). It monitors the propagated `SWEEP_CONFIG.monitor` (schema default `val_mean_dice_per_image`; `val_best_dice_at_threshold` in the shipped config) — do not set `monitor` under `lr_scheduler_config`. Configured via `lr_scheduler_config`.
  - `"cosine_annealing"`: `CosineAnnealingLR` with `T_max` (defaults to `max_epochs` when `None`) and `eta_min` (default `1e-6`). Configured via `cosine_annealing_config`.

### Mixed precision and gradient scale monitoring

`on_before_optimizer_step()` reads `trainer.precision_plugin.scaler` and logs `grad_scale` each step. A sudden drop or collapse in `grad_scale` is the first diagnostic signal for mixed-precision instability.

### Non-finite detection

Every training, validation, and test step validates inputs, logits, masks, and loss for NaN/Inf. On detection a detailed `_tensor_debug_summary()` is raised with the batch index and per-tensor statistics.

## Model architecture

SkiNet uses a UNet2D architecture configured via {py:class}`SkiNet.ML.configs.model_configs.unet2d_config.UNet2DModelConfig` (under `MODEL_CONFIG` in the YAML).

### Key UNet2DModelConfig fields

| Field | Default | Description |
|---|---|---|
| `in_channels` | `3` | Input image channels |
| `out_channels_layer1` | `16` | Output channels of the first encoder layer |
| `number_of_layers` | `5` | Total encoder layers (decoder has `number_of_layers - 1` + one final conv) |
| `num_output_classes` | `1` | Segmentation output classes |
| `kernel` | `3` | Convolution kernel size |
| `stride` | `2` | Downsampling stride in encoder; upsampling factor in decoder |
| `encoder_residual_mode` | `"he2"` | Residual block type for encoder layers |
| `merge_residual_mode` | `"he2"` | Residual block type for decoder merge layers |
| `se_reduction` | `16` | Channel reduction ratio for SE blocks (only used when `encoder_residual_mode="se"`) |
| `validate_forward` | `True` | Structural validation (skip key count) during forward pass |
| `debug_forward` | `False` | Log warnings for near-zero skip connections (GPU-expensive; keep off in production) |

### Encoder residual modes

Configured via `encoder_residual_mode` in `MODEL_CONFIG`:

| Mode | Description |
|---|---|
| `"classical"` | Standard UNet encoder: Conv-BN-Act without any residual connection |
| `"local_refinement"` | Post-activation: downsample → refine with residual from downsampled intermediate (Oktay et al. 2020) |
| `"he2"` (default) | Pre-activation with 1×1 projection shortcut: BN-Act-Conv → BN-Act-Conv + P(x) (He et al. ECCV 2016) |
| `"se"` | Pre-activation He2 with Squeeze-and-Excitation channel attention applied before the shortcut addition (Hu et al. CVPR 2018) |

### Merge (decoder) residual modes

Configured via `merge_residual_mode` in `MODEL_CONFIG`:

| Mode | Description |
|---|---|
| `"classical"` | Standard UNet decoder: upsample → concatenate skip → Conv-BN-Act without any residual connection |
| `"local_refinement"` | Post-activation: project-and-sum → BN-Act → Conv-BN-Act + residual (Oktay et al. 2020) |
| `"he1"` | Pre-activation with one refinement conv + identity shortcut (He et al. ECCV 2016) |
| `"he2"` (default) | Pre-activation with two refinement convs + identity shortcut (He et al. ECCV 2016) |
| `"attention_gate"` | Additive attention gate (Oktay et al. MIDL 2018) gates the skip connection before merge; post-merge he2 refinement |

Example YAML:
```yaml
MODEL_CONFIG:
  encoder_residual_mode: "he2"
  merge_residual_mode: "attention_gate"
```

## Best threshold selection

At the end of each validation epoch, **when `optimal_threshold` is `null`**, {py:func}`SkiNet.ML.training.training_utils.find_best_threshold` sweeps 51 evenly-spaced candidate values from 1.0 down to 0.0 using `torch.linspace`. For every threshold it computes true positives, false positives, and false negatives in a single vectorised broadcast across the entire validation set, then derives Dice (F1) as `2·tp / (2·tp + fp + fn)`. The threshold with the highest Dice is selected; when multiple thresholds tie, the **highest** one wins because the sweep is descending and `argmax` returns the first occurrence.

When `optimal_threshold` is set to a fixed float (as in the shipped `main_config.yaml`, `0.5`), the sweep is **skipped**: `val_best_dice_at_threshold` and `val_mean_dice_per_image` are computed directly at that fixed threshold and `val_optimal_threshold` echoes it. In both cases `val_dice_threshold_gain` reports the Dice difference relative to a plain 0.5 cutoff. `val_best_dice_at_threshold` is the metric monitored by early stopping and Optuna.

## Callbacks

All callbacks are opt-in via boolean flags in `TrainConfig`. They are wired together in
{py:func}`SkiNet.Utils.logging.logging_callbacks_setup.setup_logging_and_callbacks`.

| Callback | Flag | Description |
|---|---|---|
| {py:class}`SkiNet.Utils.logging.system_metrics.SystemMetricsThreadCallback` | always on | Background thread logs CPU%, RAM%, GPU memory (allocated/reserved), and GPU utilisation every `system_metrics_interval_sec` (default 5 s) |
| {py:class}`SkiNet.Utils.logging.throughput.ThroughputCallback` | always on | Logs `perf/samples_per_sec` and `perf/time_per_step_ms` after every training batch; primary signal for GPU saturation and DataLoader bottleneck diagnosis |
| `EarlyStopping` | `use_early_stopping` | Monitors the propagated `SWEEP_CONFIG.monitor` (mode `"max"`); config via {py:class}`SkiNet.ML.configs.train_configs.train_config.EarlyStoppingConfig` (schema default patience 5; `main_config.yaml` uses 30) |
| `ModelCheckpoint` | `use_checkpoint` | Saves the best checkpoint by the propagated `SWEEP_CONFIG.monitor`; config via {py:class}`SkiNet.ML.configs.train_configs.train_config.CheckpointConfig` |
| `MLFlowLogger` | `use_mlflow_logger` | Logs params, metrics, model summary, and artifacts; supports nested Optuna child runs; config via {py:class}`SkiNet.ML.configs.train_configs.train_config.MLflowConfig` |
| `LearningRateMonitor` | any logger enabled | Logs LR each epoch |
| {py:class}`SkiNet.Utils.mlops.mlflow_callbacks.MLflowTrainingArtifactsCallback` | `use_mlflow_logger` | Logs model summary at fit start; logs early-stopping state and best checkpoint as artifacts at fit end |
| `LitLogger` | `use_litlogger_logger` | Lightning Studio native logger; config via {py:class}`SkiNet.ML.configs.train_configs.train_config.LitLoggerConfig` |

## Ways to start training inside a configured environment (Docker container)

The options below assume you are inside a configured environment (as per SkiNet's Docker container)

### Optuna hyperparameter optimisation (HPO) sweep

When running `optuna_sweep.py`, a single MLflow parent run wraps the whole study. Each trial is a nested MLflow child run. The sampler is `GridSampler` (exhaustive grid, not random).

- The search space is declared in {py:class}`SkiNet.ML.configs.train_configs.sweep_config.SweepConfig` and keyed by {py:class}`SkiNet.Utils.experiment_keys.HyperparamKey`.
  `HyperparamKey` is the single source of truth: adding a new member there automatically makes
  {py:func}`SkiNet.Utils.mlops.optuna_utils.validate_search_space` require it and allows `build_objective` to read it.

  Current members:

  | Member | String key | Description |
  |---|---|---|
  | `HyperparamKey.LR` | `"lr"` | Learning rate (scaled linearly by batch size: `lr * batch_size / min_batch_size`). Fixed to a single value in `main_config.yaml` after E1 LR search; set multiple values in `SWEEP_CONFIG.lr` only when searching LR for a new architecture. |
  | `HyperparamKey.WEIGHT_DECAY` | `"weight_decay"` | L2 regularisation strength |
  | `HyperparamKey.BATCH_SIZE` | `"batch_size"` | Samples per training batch (also rescales LR — see below) |
  | `HyperparamKey.NUM_WORKERS` | `"num_workers"` | DataLoader worker count |
  | `HyperparamKey.PREFETCH_FACTOR` | `"prefetch_factor"` | Batches pre-loaded per worker |
  | `HyperparamKey.SCHEDULER_TYPE` | `"scheduler_type"` | LR scheduler per trial: `"none"` (sets `use_lr_scheduler=False`), `"cosine_annealing"`, or `"reduce_on_plateau"` |

  Each field is a **list of GridSampler candidates** for one dimension; the effective search space is
  the Cartesian product of the lists, and `optuna_sweep.py` runs the full product
  (`n_combos = ∏ len(list)`) unless `--trials N` caps it. Single-element lists are held constant.

  Every `SweepConfig` field defaults to a **single value, kept consistent with the `SWEEP_CONFIG`
  block in `main_config.yaml`** (`lr=[3e-4]`, `weight_decay=[0.0]`, `batch_size=[8]`, `num_workers=[2]`,
  `prefetch_factor=[4]`, `scheduler_type=["none"]`), so the default grid is a **1-combo no-op sweep**.
  To search a dimension, widen its list in the YAML — in practice vary **one** at a time. Note
  `num_workers` and `prefetch_factor` are throughput knobs that do not affect model quality, and
  `batch_size` is usually swept to find the largest size that fits GPU memory rather than as a
  generalisation target. When `batch_size` is varied, `lr` is rescaled per trial by
  `scale_lr` (anchored to the smallest batch in the sweep), so a sampled `lr` always denotes the rate
  at the reference batch size.

- **`SWEEP_CONFIG.monitor` and `SWEEP_CONFIG.direction` in the YAML are the single source of truth** for the
  optimisation objective. {py:class}`SkiNet.ML.configs.experiment_config.ExperimentConfig` automatically propagates `monitor` into
  `early_stopping_config`, `checkpoint_config`, and `lr_scheduler_config` at config-load time —
  do **not** set `monitor` in those sub-sections. If they disagree, a `ValueError` is raised at
  startup so the mismatch is caught before any training runs. The schema default (when
  `SWEEP_CONFIG.monitor` is unset) is `val_mean_dice_per_image`; the shipped `main_config.yaml`
  overrides it to `val_best_dice_at_threshold`, which is what the callbacks below then monitor.

  ```yaml
  SWEEP_CONFIG:
    monitor: "val_best_dice_at_threshold"   # <-- only place to set the metric
    direction: "maximize"
    # early_stopping_config, checkpoint_config, lr_scheduler_config: omit 'monitor' — propagated automatically
  ```

- `--monitor` / `--direction` CLI flags are **optional overrides** — use them only for a one-off run
  without editing the YAML. When omitted, `optuna_sweep.py` reads directly from `SWEEP_CONFIG`.

- Similarly, the shell env vars `SWEEP_MONITOR` / `SWEEP_DIRECTION` in `on_start_gpu.sh` are optional
  overrides; leaving them unset means the YAML values are used.

- MLflow run naming:
  - Parent: `optuna_study_{TRAIN_CONFIG.experiment_name}_{monitor}`
  - Child: `trial_{n}_lr{lr}_wd{weight_decay}_bs{batch_size}_nw{num_workers}_pf{prefetch_factor}_sched{scheduler_type}`

Example — monitor and direction taken from `SWEEP_CONFIG` in the YAML (recommended):
```bash
python optuna_sweep.py --config main_config.yaml
```
Example — one-off override without editing the YAML:
```bash
python optuna_sweep.py --config main_config.yaml --monitor val_best_dice_at_threshold --trials 10
```

### Regular training

```bash
python main_run.py --config main_config.yaml
```

MLflow run name: `{experiment_name}_seed{seed}_{timestamp}`.

### Multi-seed training

- Launch a training experiment as specified in TRAIN_CONFIG, using multiple seeds. All seeds will run under one MLflow experiment

```bash
python run_seeds.py --config main_config.yaml --seeds 42 200 300
```

Each seed produces an independent MLflow run.

## Launching training from Lightning Studio

The startup scripts (`on_start_gpu.sh`, `on_start_cpu.sh`) handle Docker bootstrap on Lightning Studio and dispatch to the commands above. See [development.md](development.md#lightning-studio) for the full reference: available `MODE` values, `DATASET`/`ENCODER_MODES`/`MERGE_MODES`/`RELEASE_GPU` env vars, dry-run, and example invocations.

## Training monitoring

### MLFlow

**When using `on_start_gpu.sh` / `on_start_cpu.sh`:** MLflow is started automatically by the script — no manual step needed.

**When running Python entry points directly inside the container:** start the MLflow server first.

Via the setup script (recommended):
```bash
chmod +x start_mlflow.sh
./start_mlflow.sh
```

Or manually (SQLite backend + local artifact store):
```bash
mlflow server \
  --backend-store-uri sqlite:////workplace/SkiNet/mlflow.db \
  --default-artifact-root file:///workplace/SkiNet/mlruns \
  --host 0.0.0.0 \
  --port 5000
```

Open the MLflow UI in a browser on port 5000. If using Lightning Studio, tunnel via SSH:
```bash
ssh -N -L 5000:localhost:5000 ssh_connection_string_from_your_studio@ssh.lightning.ai
```

### GPU utilisation

```bash
nvidia-smi dmon -s u
```

## Reproducibility

### How the same seed value is guaranteed

Three independent seed fields are read from the YAML and, in the shipped `main_config.yaml`, set to the
same integer `100` (Pydantic schema defaults differ: `split_random_seed=42`, `seed_value=None`, `seed=42`):

```python
DATA_CONFIG.split_random_seed   # used once when datasets are split
TRANSFORM_CONFIG.seed_value     # passed into the augmentation pipeline constructor
TRAIN_CONFIG.seed               # applied globally
```

Then in `configure_reproducibility()`:

```python
L.seed_everything(train_cfg.seed, workers=True)  # seeds Python, NumPy, PyTorch, and DataLoader workers
# fallback: if TRANSFORM_CONFIG.seed_value is None (and compose_kwargs has no "seed"), set it to train_cfg.seed
```

All three fields are set to a constant in the YAML — nothing overrides them at runtime.

> **Note:** Toggling `train_and_evaluate(..., visualize=...)` (default `True`, which calls
> `visualize_augmented_data` on the train split) does **not** change model weight initialisation:
> `visualize_augmented_data` snapshots torch/NumPy/Python/CUDA RNG state and restores it in a
> `finally` block, so the RNG seen by downstream model init and training is identical with or without
> visualisation. No manual re-seeding is needed.

### Platform notes — Deterministic mode (Apple Silicon / MPS)

Lightning raises `MisconfigurationException` if `deterministic: true` is used on Apple Silicon (MPS) because the MPS backend does not support deterministic mode. This is handled automatically:

```python
# On MPS, fall back to "warn" so Lightning logs a warning but continues.
# True determinism requires CUDA/cuDNN.
if train_cfg.deterministic and torch.backends.mps.is_available():
    deterministic = "warn"
else:
    deterministic = train_cfg.deterministic

if deterministic is True and torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

```yaml
trainconfig:
  deterministic: true
```

On Ubuntu/CUDA (the primary SkiNet target) this fully enables cuDNN deterministic mode.
