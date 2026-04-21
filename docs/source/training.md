# Training

## Training monitoring

### MLFlow

- Before running a traning script, start MLflow server using either of methods below:
- Start MLflow tracking server either using this command (SQLite backend + local artifact store):
```bash
mlflow server \
  --backend-store-uri sqlite:////workplace/SkiNet/mlflow.db \
  --default-artifact-root file:///workplace/SkiNet/mlruns \
  --host 0.0.0.0 \
  --port 5000
```

- Start running a setup script:
```bash
chmod +x start_mlflow.sh
./start_mlflow.sh
```

- Open MLflow UI in browser on port 5000. If using Lightning studio, ssh to it via
```bash
ssh -N -L 5000:localhost:5000 ssh_connection_string_from_your_studio@ssh.lightning.ai
```

## Ways to start training

The options below assume you are inside a configured environment

### Optuna hyperparameter optimisation (HPO) sweep

When running `optuna_sweep.py`, a single MLflow parent run wraps the whole study that consists of multiple child runs (trials),
where the hyperparameters of each trial are sampled from the search space.

- The search space is specified in the main function (can be moved to a config later)
- The metrics that is monitored by Optuna and displayed as the best at the end of the sweep is specified under ```--monitor``` argument


Example using default number of trials:
    ```python
    python optuna_sweep.py --config main_config.yaml --monitor val_dice --direction maximize
    ```
Example using a custom number of trials:
    ```python
    python optuna_sweep.py --config main_config.yaml --monitor val_dice --direction maximize --trials 10
    ```

### Regular training

- Regular training is started using user-provided YAML config file:
```bash
python main_run.py --config main_config.yaml
```

## GPU training on Lightning

- The following command will set up the docker environment, run MLFlow and and Optuna hyperparameter sweep
Example:

```bash
RUN_TRAINING=true MODE=sweep bash on_start_gpu.sh
```

- For a regular training on GPU without any Optuna sweep, the following will set up the docker environment, run MLFlow and the regular training
Example:

```bash
RUN_TRAINING=true MODE=train bash on_start_gpu.sh
```



## Reproducibility

### How the same seed value is guaranteed

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

### Platform notes — Deterministic mode (Apple Silicon / MPS)

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

