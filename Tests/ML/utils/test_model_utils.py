import pytest

from SkiNet.ML.utils.model_utils import MLWorkflowState, state_mapping


@pytest.mark.parametrize("input_state, expected_state", [
    ("train", MLWorkflowState.TRAIN),
    ("training", MLWorkflowState.TRAIN),
    ("val", MLWorkflowState.VAL),
    ("validation", MLWorkflowState.VAL),
    ("test", MLWorkflowState.TEST),
    ("testing", MLWorkflowState.TEST),
])
def test_state_mapping_valid_states(input_state, expected_state):
    """
    Test that valid state names are correctly mapped to MLWorkflowState in state_mapping
    """
    assert state_mapping(input_state) == expected_state


@pytest.mark.parametrize("input_state, expected_state", [
    ("TRAIN", MLWorkflowState.TRAIN),
    ("Training", MLWorkflowState.TRAIN),
    ("VAL", MLWorkflowState.VAL),
    ("Validation", MLWorkflowState.VAL),
    ("TEST", MLWorkflowState.TEST),
    ("Testing", MLWorkflowState.TEST),
])
def test_state_mapping_case_insensitivity(input_state, expected_state):
    """
    Test that state names are case-insensitive in state_mapping
    """
    assert state_mapping(input_state) == expected_state


def test_invalid_state_inputs():
    """
    Test that invalid state names raise appropriate errors in state_mapping
    """
    invalid_states = [
        "",           # Empty string
        "invalid",    # Invalid state name
        "trainx",     # Similar but invalid
        "valx",       # Similar but invalid
        "testx",      # Similar but invalid
        "123",        # Numbers
        None,         # None value
    ]

    for invalid_state in invalid_states:
        with pytest.raises(ValueError, match="Invalid ML workflow state name"):
            state_mapping(invalid_state)
