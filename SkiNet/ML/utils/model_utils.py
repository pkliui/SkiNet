from enum import Enum, unique


@unique
class MLWorkflowState(Enum):
    """
    Keeps ML workflow states
    """
    TRAIN = "Training"
    VAL = "Validation"
    TEST = "Testing"


def state_mapping(input_state_name: str) -> MLWorkflowState:
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

    if input_state_name.lower() in state_mapping_dict.keys():
        state = state_mapping_dict[input_state_name.lower()]
    else:
        raise ValueError(f"Invalid ML workflow state name: '{input_state_name}'. Must be one of {list(state_mapping_dict.keys())}.")
    return state