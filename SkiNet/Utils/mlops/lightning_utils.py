
from typing import Literal

import torch
import lightning as L
from SkiNet.ML.configs.experiment_config import ExperimentConfig


def configure_reproducibility(main_config: ExperimentConfig) -> bool | Literal['warn']:
    """
    Apply a single training seed across Lightning, PyTorch, and transform config defaults.
    """
    train_cfg = main_config.trainconfig
    L.seed_everything(train_cfg.seed, workers=True)

    # MPS does not support deterministic=True in Lightning; use "warn" to continue.

    deterministic: bool | Literal['warn']  # for mypy
    if train_cfg.deterministic and torch.backends.mps.is_available():
        deterministic = "warn"  # use "warn" to avoid MisconfigurationException on MPS
    else:
        deterministic = train_cfg.deterministic

    if deterministic is True and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # If transform seed is unset, inherit from TRAIN_CONFIG.seed.
    if main_config.transformconfig.seed_value is None and "seed" not in main_config.transformconfig.compose_kwargs:
        main_config.transformconfig.seed_value = train_cfg.seed
    return deterministic
