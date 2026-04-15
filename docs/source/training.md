# Cheatsheet

**See ports busy with non-docker processes**
- Install lsof
```bash
sudo apt-get update
sudo apt-get install lsof
```
- Example: port 6006
```bash
lsof -iTCP:6006 -sTCP:LISTEN -n -P
```

- Kill the process
```bash
kill <PID>
```

Start MLflow tracking server either using this command (SQLite backend + local artifact store):
```bash
mlflow server \
  --backend-store-uri sqlite:////workplace/SkiNet/mlflow.db \
  --default-artifact-root file:///workplace/SkiNet/mlruns \
  --host 0.0.0.0 \
  --port 5000
```

or running a setup script:
```bash
chmod +x start_mlflow.sh
./start_mlflow.sh
```

Open MLflow UI in browser on port 5000. If using Lightning studio, ssh to it via
```bash
ssh -N -L 5000:localhost:5000 ssh_connection_string_from_your_studio@ssh.lightning.ai
```


## Lightning + MLflow notes

With `TRAIN_CONFIG.use_mlflow_logger: true`, runs now capture:
- Training/validation metrics (`train_loss`, `val_loss`, `train_accuracy`, `val_accuracy`) and test metrics (`test_loss`, `test_accuracy`, `average_test_accuracy`) when `run_test_after_fit: true`.
- Fit + optimizer params such as batch size, max epochs, optimizer name, learning rate, and epsilon.
- Model summary artifact at fit start (`model/model_summary.txt`) when `mlflow_log_model_summary: true`.
- Lightning model artifact at fit end via `mlflow_log_model`.
- Early stopping params/state when `use_early_stopping: true` (for example `patience`, `min_delta`, `best_score`, `wait_count`, `stopped_epoch`, and `triggered`).
- Best checkpoint artifact under `checkpoints/best` when early stopping triggers and a best model path is available.


# Reproducibility

## How the same seed value is guaranteed

- Seed values don't share a live RNG state — they each get the same integer value (100) configured independently in the main configuration YAML file:

```python
main_config = load_config_from_yaml(cfg_path)
```

At this point, from the YAML:
```python
DATA_CONFIG.split_random_seed = 100 # goes into dataconfig
TRANSFORM_CONFIG.seed_value = 100 # goes into transformconfig
TRAIN_CONFIG.seed = 100 # goes into trainconfig
```
- The split seed is used once when datasets are created
- The transform seed is passed into the transform pipeline constructor
- Then in configure_reproducibility(), the train seed is applied globally via seed_everything:
```python
L.seed_everything(train_cfg.seed, workers=True)  # seeds global RNG with 100
# fallback: if transform seed_value is None, set it to train_cfg.seed (100)
```

- **All three fields are set to a constant value in the YAML, and nothing overrides them at runtime.**

- **Note:** A run with visualize=True and visualize=False will produce different model weight initialisations. The fix to re-seed after visualisation.

## Platform notes — Deterministic mode (Apple Silicon / MPS)

- Lightning raises a MisconfigurationException if `deterministic: true` is used on Apple Silicon (MPS) because MPS does not support deterministic mode.
- If any developer uses a Mac, Trainer construction will crash unless the config or code handles this case.

How it is handled in code:

```python
# On Apple Silicon (MPS) Lightning raises MisconfigurationException if deterministic=True
# because the MPS backend doesn't support deterministic mode. To avoid crashing at Trainer
# construction when a developer runs on a Mac:
#  - Preferred: set deterministic = "warn" so Lightning logs a warning but continues.
#  - Alternative: set deterministic = False to silently disable determinism.
# Note: True deterministic behavior requires CUDA/cuDNN. We only set cuDNN flags when CUDA is available.
import torch
if train_cfg.deterministic and torch.backends.mps.is_available():
    deterministic = "warn"  # use "warn" to avoid MisconfigurationException on MPS
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

- For training on Ubuntu, as in SkiNet, this should not be a problem.
