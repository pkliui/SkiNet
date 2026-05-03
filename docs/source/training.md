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
| `cache_in_ram` | `True` | Cache dataset in RAM before training |
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

## Lightning model

The core training logic lives in {py:class}`SkiNet.ML.model.lightning_model.LightningModel` (a `L.LightningModule` subclass).

### Metrics

| Metric logged | Description |
|---|---|
| `train_dice` / `val_dice` / `test_dice` | `BinaryF1Score` at the fixed or swept threshold |
| `train_iou` / `val_iou` / `test_iou` | `BinaryJaccardIndex` |
| `val_best_dice_at_threshold` | Best Dice found across the threshold sweep |
| `val_optimal_threshold` | Threshold that achieved `val_best_dice_at_threshold` |
| `val_dice_threshold_gain` | Dice gain from using the optimal threshold vs. 0.5 |
| `val_threshold_used` | Threshold actually applied during validation |

### Optimizer and scheduler

- **Optimizer:** Adam or AdamW (set via `optimizer_name`), with configurable `lr` and `weight_decay`.
- **Scheduler:** `ReduceLROnPlateau` monitors `val_best_dice_at_threshold` (mode `"max"`, patience 5, factor 0.5).

### Mixed precision and gradient scale monitoring

`on_before_optimizer_step()` reads `trainer.precision_plugin.scaler` and logs `grad_scale` each step. A sudden drop or collapse in `grad_scale` is the first diagnostic signal for mixed-precision instability.

### Non-finite detection

Every training, validation, and test step validates inputs, logits, masks, and loss for NaN/Inf. On detection a detailed `_tensor_debug_summary()` is raised with the batch index and per-tensor statistics.

## Best threshold selection

At the end of each validation epoch, `find_best_threshold` sweeps 51 evenly-spaced candidate values from 1.0 down to 0.0 using `torch.linspace`. For every threshold it computes true positives, false positives, and false negatives in a single vectorised broadcast across the entire validation set, then derives Dice (F1) as `2·tp / (2·tp + fp + fn)`. The threshold with the highest Dice is selected; when multiple thresholds tie, the **highest** one wins because the sweep is descending and `argmax` returns the first occurrence. `val_best_dice_at_threshold` is the metric monitored by early stopping and Optuna.

## Callbacks

All callbacks are opt-in via boolean flags in `TrainConfig`. They are wired together in
{py:func}`SkiNet.Utils.logging.logging_callbacks_setup.setup_logging_and_callbacks`.

| Callback | Flag | Description |
|---|---|---|
| `SystemMetricsThreadCallback` | always on | Background thread logs CPU%, RAM%, GPU memory (allocated/reserved), and GPU utilisation every `system_metrics_interval_sec` (default 5 s) |
| `EarlyStopping` | `use_early_stopping` | Monitors `val_best_dice_at_threshold` (mode `"max"`, patience 5); config via `early_stopping_config` |
| `ModelCheckpoint` | `use_checkpoint` | Saves best checkpoint by `val_best_dice_at_threshold`; config via `checkpoint_config` |
| `MLFlowLogger` | `use_mlflow_logger` | Logs params, metrics, model summary, and artifacts; supports nested Optuna child runs; config via `mlflow_config` |
| `LearningRateMonitor` | any logger enabled | Logs LR each epoch |
| `MLflowTrainingArtifactsCallback` | `use_mlflow_logger` | Logs model summary at fit start; logs early-stopping state and best checkpoint as artifacts at fit end |
| `LitLogger` | `use_litlogger_logger` | Lightning Studio native logger; config via `litlogger_config` |

## Ways to start training inside a configured environment (Docker container)

The options below assume you are inside a configured environment (as per SkiNet's Docker container)

### Optuna hyperparameter optimisation (HPO) sweep

When running `optuna_sweep.py`, a single MLflow parent run wraps the whole study. Each trial is a nested MLflow child run. The sampler is `GridSampler` (exhaustive grid, not random).

- The search space is declared in `SweepConfig` and keyed by `HyperparamKey` (`SkiNet/Utils/experiment_keys.py`).
  `HyperparamKey` is the single source of truth: adding a new member there automatically makes
  `validate_search_space` require it and allows `build_objective` to read it.

  Current members:

  | Member | String key | Description |
  |---|---|---|
  | `HyperparamKey.LR` | `"lr"` | Base learning rate (scaled linearly by batch size: `lr * batch_size / min_batch_size`) |
  | `HyperparamKey.WEIGHT_DECAY` | `"weight_decay"` | L2 regularisation strength |
  | `HyperparamKey.BATCH_SIZE` | `"batch_size"` | Number of samples per training batch |

  Default grid in `SweepConfig`: `lr ∈ [3e-4, 1e-4]`, `weight_decay ∈ [1e-4, 1e-3]`, `batch_size ∈ [16, 32]` → 8 trials.

- The metric monitored by Optuna is set via `--monitor` (default `"val_best_dice_at_threshold"`).
- The same metric **must** be specified in the YAML under `early_stopping_config`:
```yaml
early_stopping_config:
    monitor: "val_best_dice_at_threshold"
```
- MLflow run naming:
  - Parent: `optuna_study_{experiment_name}_{monitor}`
  - Child: `trial_{n}_lr{lr}_wd{weight_decay}_bs{batch_size}`

Example using default number of trials:
```bash
python optuna_sweep.py --config main_config.yaml --monitor val_best_dice_at_threshold --direction maximize
```
Example with a custom number of trials:
```bash
python optuna_sweep.py --config main_config.yaml --monitor val_best_dice_at_threshold --direction maximize --trials 10
```

### Regular training

```bash
python main_run.py --config main_config.yaml
```

MLflow run name: `{experiment_name}_seed{seed}_{timestamp}`.

### Multi-seed training

```bash
python run_seeds.py --config main_config.yaml --seeds 42 200 300
```

Each seed produces an independent MLflow run.

## GPU training on Lightning Studio

`on_start_gpu.sh` clones/updates the repo, pulls the Docker image (`pkliui/skinet:v9gpu`), mounts the repo and data, starts MLflow, and then dispatches based on `MODE`:

| MODE | What runs |
|---|---|
| `train` | `python main_run.py --config main_config.yaml` |
| `seeds` | `python run_seeds.py --config main_config.yaml --seeds <SEEDS>` |
| `sweep` | `python optuna_sweep.py --config main_config.yaml --monitor val_dice --direction maximize` |

On success the container is cleaned up and the studio switches to CPU. On failure the container is kept for debugging.

```bash
# Optuna sweep
RUN_TRAINING=true MODE=sweep bash on_start_gpu.sh

# Regular training
RUN_TRAINING=true MODE=train bash on_start_gpu.sh

# Multi-seed training
RUN_TRAINING=true MODE=seeds SEEDS="42 200 300" bash on_start_gpu.sh
```

## Training monitoring

### MLFlow

- Before running a training script, start the MLflow server using either of the methods below.
- Start MLflow tracking server using this command (SQLite backend + local artifact store):
```bash
mlflow server \
  --backend-store-uri sqlite:////workplace/SkiNet/mlflow.db \
  --default-artifact-root file:///workplace/SkiNet/mlruns \
  --host 0.0.0.0 \
  --port 5000
```

- Or start it via the setup script:
```bash
chmod +x start_mlflow.sh
./start_mlflow.sh
```

- Open the MLflow UI in a browser on port 5000. If using Lightning Studio, tunnel via SSH:
```bash
ssh -N -L 5000:localhost:5000 ssh_connection_string_from_your_studio@ssh.lightning.ai
```

### GPU utilisation

```bash
nvidia-smi dmon -s u
```

## Reproducibility

### How the same seed value is guaranteed

Three independent seed fields are read from the YAML and set to the same integer (default 100):

```python
DATA_CONFIG.split_random_seed   # used once when datasets are split
TRANSFORM_CONFIG.seed_value     # passed into the augmentation pipeline constructor
TRAIN_CONFIG.seed               # applied globally
```

Then in `configure_reproducibility()`:

```python
L.seed_everything(train_cfg.seed, workers=True)  # seeds Python, NumPy, PyTorch, and DataLoader workers
# fallback: if TRANSFORM_CONFIG.seed_value is None, set it to train_cfg.seed
```

All three fields are set to a constant in the YAML — nothing overrides them at runtime.

> **Note:** A run with `visualize=True` and one with `visualize=False` will produce different model weight initialisations. Re-seed after visualisation to avoid this.

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
