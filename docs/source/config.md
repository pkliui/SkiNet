# Loading experiment configurations

Most users interact with configurations via the YAML loader:

```python
from pathlib import Path
from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml

cfg = load_config_from_yaml(Path("experiments/my_experiment.yaml"))
# cfg is an ExperimentConfig (pydantic model) with attrs:
# cfg.experiment_type, cfg.experiment_name, cfg.description,
# cfg.dataconfig, cfg.modelconfig, cfg.trainconfig
```

YAML structure (minimal):
```yaml
GENERAL_CONFIG:
  experiment_type: "segmentation"
  model: "unet2d_model"
  dataset: "ph2_dataset"
DATA_CONFIG:
  local_data_root: "ph2data/"
  azure_data: False
MODEL_CONFIG:
TRAIN_CONFIG:

```


Notes:
- The loader validates presence of required top-level keys and allowed experiment types.
- If you add a new model or dataset, a corresponding factory/creator must be registered so the loader can produce the proper ExperimentConfig.
- For development iteration: data/train sections are currently tolerant of extra keys; modelconfig is strict to catch typos.