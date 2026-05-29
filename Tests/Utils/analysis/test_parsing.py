import pytest

from SkiNet.Utils.analysis.parsing import parse_encoder_merge


@pytest.mark.parametrize(
    ("experiment_name", "expected"),
    [
        (
            "the-model-sweep-over-merge-2gpus-aug-adam_enc-classical_merge-he2",
            ("classical", "he2"),
        ),
        (
            "the-model-sweep-over-merge-2gpus-aug-adam_enc-local_refinement_merge-attention_gate",
            ("local_refinement", "attention_gate"),
        ),
        (
            "the-model-sweep-over-merge-2gpus-aug-adam_enc-he2_merge-he2_characterisation",
            ("he2", "he2_characterisation"),
        ),
    ],
)
def test_parse_encoder_merge_returns_encoder_and_merge(
    experiment_name: str, expected: tuple[str, str]
) -> None:
    assert parse_encoder_merge(experiment_name) == expected


@pytest.mark.parametrize(
    "experiment_name",
    [
        "unrelated_experiment",
        "the-model-sweep-over-merge-2gpus-aug-adam_enc-classical",
        "the-model-sweep-over-merge-2gpus-aug-adam_merge-he2_enc-classical",
        "the-model-sweep-over-merge-2gpus-aug-adam_enc-classical-merge-he2",
    ],
)
def test_parse_encoder_merge_returns_none_when_pattern_is_missing(
    experiment_name: str,
) -> None:
    assert parse_encoder_merge(experiment_name) == (None, None)
