# Debugging

## Debug configuration

When debugging on Lightning Studio (not inside Docker), edit `debug_main_config.yaml`:
- Change `azure_blob_mount_point: "/mnt/data/"` to `azure_blob_mount_point: "/teamspace/lightning_storage/"`
- Set `local_data_root` to the actual data path on the studio

In VSCode:
1. Go to "Run and Debug" and select "Current python file with arguments"
2. Click the settings wheel icon to open `launch.json`
3. Set `"args": "--config ~/repos/SkiNet/debug_main_config.yaml"`
4. Ensure you are in the repo root, then click Run in Debugger

---

## NaN / loss divergence

### Symptoms
`val_best_dice_at_threshold` collapses to 0.0 or NaN. `grad_scale` logged by
`on_before_optimizer_step` drops suddenly to a very small value or 0.

### Causes and fixes

**1. Mixed precision instability**

`LightningModel.on_before_optimizer_step` reads `trainer.precision_plugin.scaler` and logs
`grad_scale` each step. A sustained drop in `grad_scale` (e.g. from 32768 to 1) signals
that gradients are overflowing in fp16.

Fix: override precision to full:
```yaml
TRAIN_CONFIG:
  precision: "32-true"
```

**2. NaN in input/loss detected by non-finite detection**

Every training, validation, and test step guards inputs, masks, logits, and loss with
`_raise_if_non_finite`, which raises a `ValueError` embedding `_tensor_debug_summary` output
(min, max, mean, std, non-finite count) when any element is NaN or Inf. Check the error message for
which named tensor (`{prefix}/image`, `{prefix}/mask`, `{prefix}/logits`, …) triggered it and on
which batch index.

**3. Learning rate too high**

Loss spikes in the first few batches then diverges. Fix: reduce `lr` in `TRAIN_CONFIG`.
For ISIC 2017 the validated value is `lr: 3e-4` (E1 LR sweep result).

**4. `bce_dice` loss with unscaled logits**

`LightningModel` computes the loss on raw logits (numerically stable) and applies sigmoid only
for metrics/inference. Ensure no extra activation is added to the network's forward pass --
`UNet2D.forward` returns raw logits and sigmoid is applied only inside
`LightningModel._get_probs_and_preds`.

---

## GPU memory exhaustion (OOM)

### Symptoms
`CUDA out of memory` error during training or validation, or RSS memory grows every epoch.

### Batch size guidance (ISIC 2017, single T4 GPU)

| Batch size | Approx. VRAM (fp16) | Notes |
|---|---|---|
| 8 | ~1.5 GB | Optuna-selected default |
| 16 | ~2.5 GB | Viable; use `cache_in_ram: false` if host RAM is low |
| 32 | ~4.5 GB | For ablation comparisons only; not used for final training |

These values are approximate. Actual VRAM depends on input resolution and `number_of_layers`.

### Fixes

- Reduce `batch_size` in `TRAIN_CONFIG`
- Disable `cache_in_ram` if host RAM is also tight -- this does not affect VRAM
- Set `use_torch_compile: false` during debugging (compilation adds temporary VRAM overhead)
- Add `num_sanity_val_steps: 0` to skip early validation that can trigger OOM before training
- Switch from `"bce_dice"` to `"bce"` temporarily to check if Dice's intermediate tensors cause OOM

### Memory growth per epoch

If RSS grows each epoch but VRAM is stable, the cause is copy-on-write in forked worker processes
(see [dataloaders.md](dataloaders.md)). Verify that `RepeatDataLoader` is being used (not raw
`DataLoader`) and that `num_workers > 0`.

---

## DataLoader bottleneck diagnosis

`ThroughputCallback` logs `perf/samples_per_sec` and `perf/time_per_step_ms` after every training
batch. If `samples_per_sec` is low relative to GPU compute capacity, the bottleneck is data loading.

### Diagnosis steps

1. Check MLflow metrics: if `perf/time_per_step_ms` is high and GPU utilisation (from
   `SystemMetricsThreadCallback`) is low (<50%), data loading is the bottleneck.

2. Run `nvidia-smi dmon -s u` in parallel to watch GPU utilisation during training.

3. Increase `num_workers` -- default is auto-set to `os.cpu_count()`. On a 4-core machine try
   `num_workers: 4`. More workers than CPU cores hurts performance.

4. Increase `prefetch_factor` -- default is `null` (PyTorch default of 2). Try `prefetch_factor: 4`
   or `8` if augmentation is slow relative to GPU compute.

5. Enable `cache_in_ram: true` (default) for small datasets. Workers then only augment, never read
   from disk. This is the single largest DataLoader speedup for ISIC 2017.

6. Check `normalization_mode`: `"image_per_channel"` computes mean/std per sample at runtime
   (roughly 20x slower than `"standard"`). Use `"standard"` with pre-computed ISIC 2017 statistics
   for production training.

---

## Checkpoint / resume errors

### `_orig_mod.` prefix mismatch

When `use_torch_compile: true`, PyTorch wraps the model and prefixes all state dict keys with
`_orig_mod.`. A checkpoint saved with `use_torch_compile: true` cannot be loaded with
`use_torch_compile: false` and vice versa. The `main_config.yaml` comment documents this:

```yaml
TRAIN_CONFIG:
  use_torch_compile: true  # MUST match checkpoint: E4 ckpt keys carry _orig_mod. prefix
  torch_compile_backend: eager  # still wraps model so _orig_mod. keys match
```

Fix: match the `use_torch_compile` setting to the checkpoint that was saved.

### Monitor metric missing at checkpoint restore

If `ModelCheckpoint` cannot find the monitored metric in the logged metrics dict, Lightning raises
a warning and may not save. Verify that `SWEEP_CONFIG.monitor` matches exactly what
`LightningModel` logs (e.g. `val_best_dice_at_threshold`).

### Epoch counter reset after resume

Lightning resumes from the epoch stored in the checkpoint. If `max_epochs` in the config is less
than the checkpoint epoch, training exits immediately. Set `max_epochs` higher than the checkpoint
epoch to continue training.
