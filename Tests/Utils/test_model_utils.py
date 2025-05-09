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
def test_valid_states(input_state, expected_state):
    """
    Test that valid state names are correctly mapped to MLWorkflowState.
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
def test_case_insensitivity(input_state, expected_state):
    """
    Test that state names are case-insensitive.
    """
    assert state_mapping(input_state) == expected_state


@pytest.mark.parametrize("invalid_state", [
    "invalid_state",
    "",
    "   ",
    "unknown",
    "123",
])
def test_invalid_states(invalid_state):
    """
    Test that invalid state names raise a ValueError.
    """
    with pytest.raises(ValueError, match="Invalid state name"):
        state_mapping(invalid_state)