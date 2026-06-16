import inspect
import logging
import os
import random
from enum import Enum, unique
from typing import Optional, cast

import numpy as np
from torch import manual_seed
from torch.nn import BatchNorm2d, Conv2d, ConvTranspose2d, Module, init


@unique
class MLWorkflowState(Enum):
    """
    Keeps ML workflow states
    """
    TRAIN = "Training"
    VAL = "Validation"
    TEST = "Testing"


def state_mapping(input_state_name: Optional[str]) -> MLWorkflowState:
    """
    Mapping of ML workflow states to their string representations of different spellings

    :param input_state_name: The string representation of the state as provided by user, e.g. "train", "Train" or "training" for training dataset
    :return: The corresponding MLWorkflowState enum value.
    """

    #
    state_mapping_dict = {
        "train": MLWorkflowState.TRAIN,
        "training": MLWorkflowState.TRAIN,
        "val": MLWorkflowState.VAL,
        "validation": MLWorkflowState.VAL,
        "test": MLWorkflowState.TEST,
        "testing": MLWorkflowState.TEST,
    }

    if input_state_name is not None and (input_state_name.lower() in state_mapping_dict.keys()):
        state = state_mapping_dict[input_state_name.lower()]
    else:
        raise ValueError(f"Invalid ML workflow state name: '{input_state_name}'. Must be one of {list(state_mapping_dict.keys())}.")
    return state


def set_random_seed(random_seed: int, module_name: Optional[str] = None) -> None:
    """
    Seed for random number generators

    :param random_seed: seed value
    :param module_name: module name where this seed is called from
    """

    if module_name is None:
        module_name = os.path.basename(inspect.stack()[1].filename)
        logging.getLogger(__name__).debug(f"Random seed set to {random_seed} in module {module_name}.")

    manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

def initialise_weights(module: Module) -> None:
    """
    Initialise the conv layers' weights using Kaiming method as described in Delving deep into rectifiers:
    Surpassing human-level performance on ImageNet classification (2015), arXiv:1502.01852v1
    and the batchnorm's weights to a normal distribution.

    :param module: The module to initialise.

    Example usage: model.apply(initialise_weights) will apply the weight initialisation to all modules in the model.
    """
    if cast(bool, getattr(module, "_skinet_initialized", False)):
        return

    if isinstance(module, (Conv2d, ConvTranspose2d)):
        init.kaiming_normal_(module.weight, a=0, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            init.zeros_(module.bias)
        setattr(module, "_skinet_initialized", True)

    elif isinstance(module, BatchNorm2d):
        if module.affine:
            init.normal_(module.weight, mean=1.0, std=0.01)
            init.constant_(module.bias, 0.0)
            setattr(module, "_skinet_initialized", True)
