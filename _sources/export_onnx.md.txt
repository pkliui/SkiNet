# Exporting to ONNX

Trained SkiNet checkpoints can be exported to ONNX format for deployment on
mobile (iOS/Android) or any ONNX-compatible inference runtime.
The entry point is `export_onnx.py` in the repo root.

## Quick start

### From an MLflow run folder (recommended)

Point `--run` at the MLflow run directory and the script auto-discovers the
best checkpoint and the config YAML:

```bash
cd repos/SkiNet
python export_onnx.py \
    --run mlruns/<experiment>/<run_id>/<uuid> \
    --out skinet_unet.onnx
```

### Explicit paths

Supply `--ckpt` and `--config` directly when you want to target a specific
checkpoint or config that is not under `artifacts/`:

```bash
python export_onnx.py \
    --ckpt path/to/epoch=42.ckpt \
    --config path/to/config.yaml \
    --out skinet_unet.onnx \
    --opset 17
```

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--run` | `None` | MLflow run folder; checkpoint and config are auto-discovered |
| `--ckpt` | `None` | Explicit path to a `.ckpt` file (required if `--run` is omitted) |
| `--config` | `None` | Explicit path to a config YAML (required if `--run` is omitted) |
| `--out` | `skinet_unet.onnx` | Output path for the exported ONNX model |
| `--opset` | `17` | ONNX opset version |

Either `--run` **or** both `--ckpt` and `--config` must be supplied. If `--run` is given alongside `--ckpt`/`--config`, `--run` takes precedence and the explicit paths are silently ignored.

## What the script does

1. **Loads the config** via {py:func}`SkiNet.ML.configs.load_config_from_yaml.load_config_from_yaml`.
2. **Builds the model architecture** from the config with `use_torch_compile` forced to `False`
   (compilation is not needed for export and avoids environment dependencies).
3. **Loads the checkpoint** with `torch.load(..., weights_only=False)` and strips the
   `model._orig_mod.*` key prefix that `torch.compile` adds, so the state dict matches
   the uncompiled model.
4. **Reports the optimal threshold** stored in the checkpoint buffer (`optimal_threshold`).
   This value should be hard-coded as `SEGMENTATION_THRESHOLD` in the iOS/Android app.
5. **Wraps the backbone** in `_UNetWithSigmoid`, which fuses a `torch.sigmoid` into the
   ONNX graph so the model outputs probabilities (0–1) rather than raw logits.
6. **Exports via `torch.onnx.export`** with dynamic batch axis on both input and output.
7. **Merges external weight data** (if `onnx` is installed): the dynamo exporter may write
   a `.onnx.data` sidecar file; the script merges it into a single self-contained `.onnx`
   file and deletes the sidecar.
8. **Validates with ONNXRuntime** (if `onnxruntime` is installed): runs a zero-tensor
   forward pass and asserts the output shape is `(1, 1, 256, 256)`.
9. **Prints a deployment summary**: model file size, `INPUT_SIZE`, normalisation constants,
   and the optimal sigmoid threshold.

## ONNX graph

| Property | Value |
|---|---|
| Input name | `image` |
| Input shape | `(batch, 3, 256, 256)` — only the batch axis is dynamic |
| Output name | `mask_prob` |
| Output shape | `(batch, 1, 256, 256)` — probabilities in [0, 1] |
| Default opset | `17` |

```{note}
Only the batch axis is dynamic. The spatial dimensions are fixed at **256×256**, so every
input must be resized to 256×256 before inference — feeding any other height/width will fail
ONNXRuntime's shape check. 256 is the size the model is trained and deployed at.
```

## iOS / mobile preprocessing constants

The script prints these at the end of every run:

```
INPUT_SIZE  = 256
NORM_MEAN   = [0.699, 0.556, 0.5121]
NORM_STD    = [0.1576, 0.1562, 0.1706]
THRESHOLD   = <value from checkpoint>
```

Apply these in the same order as the training pipeline:
1. Resize the input image to `INPUT_SIZE × INPUT_SIZE`.
2. Normalise each channel: `(pixel / 255 − mean) / std`.
3. Run inference; apply the threshold to the probability map to get a binary mask.

## Auto-discovery rules (`--run` mode)

When `--run` is supplied, `_resolve_run` searches the run folder as follows:

- **Checkpoint:** `glob("artifacts/checkpoints/**/*.ckpt")`, sorted lexicographically.
  If any result has `"best"` in its path components, the last such path is chosen;
  otherwise the lexicographically last checkpoint overall is used.
- **Config:** `glob("artifacts/config/*.yaml")`, sorted lexicographically; the last file is used.

## Optional dependencies

| Package | Effect if missing |
|---|---|
| `onnx` | Weight-merge step is skipped; model may export as two files (`.onnx` + `.onnx.data`) |
| `onnxruntime` | Post-export shape validation is skipped |

Install both for a full export pipeline:

```bash
pip install onnx onnxruntime
```

## Checkpoint key remapping

Training may be run with `use_torch_compile: true`, which causes `torch.compile` to
prefix every state-dict key with `model._orig_mod.`. The exporter strips this prefix
automatically, so compiled and uncompiled checkpoints are both supported without any
manual key editing.
