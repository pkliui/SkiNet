import inspect
import logging
import os
import random
from enum import Enum, unique
from typing import Optional

import numpy as np
import torch


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

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
