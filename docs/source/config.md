# Configuration guide

This page explains the **concepts** behind the experiment config — how the YAML is
structured, the cross-cutting rules that span multiple fields, and the gotchas that
aren't obvious from a single field. For the exhaustive per-field reference (every
key, its type, default, allowed values, and description), see the auto-generated
**[Configs API reference](api_config.md)**, which is rendered directly from the
Pydantic models and therefore always matches the code.

All experiment settings live in a single YAML file (default: `main_config.yaml`),
loaded by `load_config_from_yaml` and validated into an `ExperimentConfig` Pydantic
model. `ExperimentConfig` and `TrainConfig` use `extra="forbid"` — unknown keys raise
a validation error, so a typo'd key fails fast rather than being silently ignored.

```python
from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml
cfg = load_config_from_yaml("main_config.yaml")
```

## YAML blocks → models

| YAML block | Parsed into | Reference |
|---|---|---|
| `GENERAL_CONFIG` | (consumed by the loader — see below) | — |
| `DATA_CONFIG` | dataset config selected by `GENERAL_CONFIG.dataset` (e.g. `ISIC2017DatasetConfig`) | [data_configs](api/data_configs.rst) |
| `TRANSFORM_CONFIG` | `TransformConfig` (+ `CropConfig`, `SpatialAugmentConfig`, `PhotoAugmentConfig`) | [transform_configs](api/transform_configs.rst) |
| `MODEL_CONFIG` | model config selected by `GENERAL_CONFIG.model` (e.g. `UNet2DModelConfig`) | [model_configs](api/model_configs.rst) |
| `SWEEP_CONFIG` | `SweepConfig` | [train_configs](api/train_configs.rst) |
| `TRAIN_CONFIG` | `TrainConfig` (+ nested callback/logger/scheduler configs) | [train_configs](api/train_configs.rst) |

---

## GENERAL_CONFIG

Unlike the other blocks, `GENERAL_CONFIG` is **not** parsed into one model. The loader
(`_validate_yaml_config`) consumes exactly three required keys and validates each at load:

- `experiment_type` — must be `"segmentation"` (the only member of `ALLOWED_EXPERIMENT_TYPES`).
- `model` — validated against `ModelKey` (e.g. `"UNET2D_MODEL"`); selects the model config class.
- `dataset` — validated against `DatasetKey` (e.g. `"ISIC2017_DATASET"`); selects the dataset
  config class and its factory.

```{warning}
`experiment_name` and `description` under `GENERAL_CONFIG` are **ignored**. The loader
reads only the three keys above. `ExperimentConfig.experiment_name` / `description` are
hardcoded by the config creator and play no part in run naming — see
[Run and experiment naming](#run-and-experiment-naming) below.
```

---

## Run and experiment naming

Naming is driven entirely by `TRAIN_CONFIG.experiment_name` and `SWEEP_CONFIG.experiment_name`,
**not** by anything under `GENERAL_CONFIG`:

- **Regular training** (`setup_logging_and_callbacks`):
  - MLflow *experiment* = `TRAIN_CONFIG.experiment_name`
  - *run* = `{TRAIN_CONFIG.experiment_name}_seed{TRAIN_CONFIG.seed}_{YYYYMMDD-HHMMSS}`
  - checkpoints → `{TRAIN_CONFIG.log_dir}/checkpoints/{run_name}`
- **Optuna sweep** (`optuna_sweep.py`):
  - MLflow *experiment* = `SWEEP_CONFIG.experiment_name` (or `--experiment`)
  - *parent run* = `optuna_study_{TRAIN_CONFIG.experiment_name}_{monitor}`
  - each *child run* = `trial_{n}_lr{lr}_wd{wd}_bs{bs}_nw{nw}_pf{pf}_sched{scheduler_type}`

---

## Cross-cutting rules

These are constraints and behaviours that span multiple fields — they can't be read off
any single field in the reference, so they live here.

### `kind` discriminators are set automatically

`DATA_CONFIG` and `MODEL_CONFIG` are discriminated unions: the `kind` field
(`"unet2d"`, the dataset kind, …) is set by the config creator from the
`GENERAL_CONFIG.model` / `dataset` keys. **Do not set `kind` in the YAML.**

### `monitor` propagates from SWEEP_CONFIG

`SWEEP_CONFIG.monitor` (and `direction`) is the single source of truth for the
optimisation objective. `ExperimentConfig` propagates `monitor` into
`TRAIN_CONFIG.early_stopping_config`, `checkpoint_config`, and `lr_scheduler_config`,
keeping all four in sync. **Do not set `monitor` in those sub-sections.**

### Sweep grids default to a no-op

Every `SWEEP_CONFIG` candidate field is a list consumed by Optuna's `GridSampler`.
Defaults are a single value per field (matching the `SWEEP_CONFIG` block in
`main_config.yaml`), so the default config is a 1-combination no-op grid. Widen one
field to a list to sweep that dimension; in practice tune one dimension at a time.

### Crop size must be divisible by the downsampling factor

`TRANSFORM_CONFIG.crop.size` (height, width) must be divisible by
`stride^(number_of_layers - 1)` for the UNet encoder/decoder shapes to line up
(see `UNet2DModelConfig.required_input_multiple`).

### `"standard"` normalization requires mean/std

When `TRANSFORM_CONFIG.normalization_mode: "standard"`, both `normalization_mean` and
`normalization_std` (per-channel RGB, in `[0, 1]`) are required. Compute them with
`compute_dataset_stats.py`. The default `"image_per_channel"` mode needs no constants
but is ~20× slower (per-sample reductions at runtime).

### DATA_CONFIG: Azure vs local, and predefined splits

- Set exactly one data source: `azure_data: true` requires `azure_blob_mount_point`;
  `azure_data: false` requires `local_data_root`.
- When `predefined_split_column` is set, rows are assigned to splits by that column's
  values and `split_train_size` / `split_val_size` / `split_test_size` are ignored.

### Auto-resolved TRAIN_CONFIG fields

A few fields default to `null` and are resolved at runtime — leave them unset unless
you need to override:

- `num_workers` → `os.cpu_count()` (single GPU) or `cpu_count // devices` (DDP).
- `pin_memory` → `True` on CUDA/GPU, `False` on MPS/CPU.
- `precision` → `"16-mixed"` on GPU/MPS, `"32-true"` on CPU.
- `prefetch_factor` → forced to `None` when `num_workers=0`.
- `cosine_annealing_config.T_max` → `max_epochs`.

### `augmentation_required` is not read

`TRANSFORM_CONFIG.augmentation_required` exists on the base transform config but is
**not** consulted by the current `get_transform_from_config` pipeline (it is only
referenced by a plotting notebook). Augmentation is controlled by the per-transform
`*_apply` flags instead.

---

## Minimal working config

`MODEL_CONFIG` and `TRANSFORM_CONFIG` may be left empty — every field has a Pydantic
default.

```yaml
GENERAL_CONFIG:
  experiment_type: "segmentation"
  model: "UNET2D_MODEL"
  dataset: "ISIC2017_DATASET"
  experiment_name: "my_experiment"   # ignored by the loader; see "Run and experiment naming"
  description: ""                     # ignored by the loader
DATA_CONFIG:
  local_data_root: "/path/to/isic2017"
  azure_data: False
TRANSFORM_CONFIG:
MODEL_CONFIG:
TRAIN_CONFIG:
  max_epochs: 10
```

For every available key and its default, see the **[Configs API reference](api_config.md)**.
