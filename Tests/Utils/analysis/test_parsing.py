from SkiNet.Utils.analysis.parsing import parse_encoder_merge
import pytest


@pytest.mark.parametrize("experiment_name,expected", [
    ("enc-classical_merge-attention_gate_lr3e-4", ("classical", "attention_gate")),
    ("enc-full_merge-he2", ("full", "he2")),
    ("enc-none_merge-none", ("none", "none")),
])
def test_parse_encoder_merge_returns_encoder_and_merge(
    experiment_name: str, expected: tuple
) -> None:
    assert parse_encoder_merge(experiment_name) == expected


@pytest.mark.parametrize("experiment_name", [
    "no_pattern_here",
    "",
    "random_string_123",
])
def test_parse_encoder_merge_returns_none_when_pattern_is_missing(
    experiment_name: str,
) -> None:
    assert parse_encoder_merge(experiment_name) == (None, None)
