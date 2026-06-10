import pytest

from SkiNet.Utils.analysis import schema


@pytest.mark.parametrize(
    ("constant", "expected"),
    [
        (schema.SEED, "seed"),
        (schema.ARCH, "arch"),
        (schema.VAL_DICE_MAX, "val_dice_max"),
        (schema.VAL_DICE_TAIL_MEAN, "val_dice_tail_mean"),
        (schema.VAL_DICE_TAIL_STD, "val_dice_tail_std"),
        (schema.VAL_IOU_MAX, "val_iou_max"),
        (schema.GENERALIZATION_GAP_FINAL, "generalization_gap_final"),
        (schema.SAMPLES_PER_SEC, "samples_per_sec"),
        (schema.DURATION_MIN, "duration_min"),
    ],
)
def test_schema_constants_match_canonical_column_names(
    constant: str, expected: str
) -> None:
    # These constants are the single source of truth for DataFrame column
    # names produced by load_runs and consumed across the analysis package
    # and notebooks. A change here silently breaks every downstream lookup,
    # so the values are pinned explicitly.
    assert constant == expected
