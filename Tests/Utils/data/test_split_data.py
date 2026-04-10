
import pytest
import pandas as pd
from SkiNet.Utils.data.split_data import stratified_split_segmentation_metadata, DataFrameSplits
from SkiNet.Utils.csv_headers import SAMPLEID_HEADER, DATATYPE_HEADER, PH2_CLINICAL_DIAGNOSIS_HEADER, DATATYPE_IMAGE, DATATYPE_MASK

NSAMPLES = 100
TRAIN_PERCENT = 0.7
VAL_PERCENT = 0.15
TEST_PERCENT = 0.15


def make_balanced_df(n_samples: int = NSAMPLES, n_classes: int = 2) -> pd.DataFrame:
    """
    Create a balanced DataFrame with the specified number of samples and classes.
    :return:
    """
    rows = []
    classes = [f"c{i}" for i in range(n_classes)]
    for i in range(n_samples):
        cid = f"s{i}"
        label = classes[i % n_classes]
        rows.append({SAMPLEID_HEADER: cid, DATATYPE_HEADER: DATATYPE_IMAGE, PH2_CLINICAL_DIAGNOSIS_HEADER: label})
        rows.append({SAMPLEID_HEADER: cid, DATATYPE_HEADER: DATATYPE_MASK, PH2_CLINICAL_DIAGNOSIS_HEADER: label})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("n_samples,n_classes", [(NSAMPLES, 2), (2*NSAMPLES, 3)])
def test_normal_split_no_overlap_and_both_types(n_samples: int, n_classes: int) -> None:
    """
    Test that the stratified split produces non-overlapping splits where each sample ID appears in only one split,
    and that each sample in each split has both image and mask datatypes.
    Also verify that the counts of samples in each split roughly match the requested proportions.
    """
    df = make_balanced_df(n_samples=n_samples, n_classes=n_classes)
    splits: DataFrameSplits = stratified_split_segmentation_metadata(
        df,
        stratify_column=PH2_CLINICAL_DIAGNOSIS_HEADER,
        train_size=TRAIN_PERCENT,
        val_size=VAL_PERCENT,
        test_size=TEST_PERCENT,
        random_seed=0)

    # no sample id appears in more than one split
    train_ids = set(splits.train[SAMPLEID_HEADER].unique())
    val_ids = set(splits.val[SAMPLEID_HEADER].unique())
    test_ids = set(splits.test[SAMPLEID_HEADER].unique())
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # each sample in each split must have both image and mask
    def check_both_types(df: pd.DataFrame) -> None:
        # group by sample id
        for sid, grp in df.groupby(SAMPLEID_HEADER):
            # check that both image and mask datatypes are present for this sample id
            types = set(grp[DATATYPE_HEADER])
            assert {DATATYPE_IMAGE, DATATYPE_MASK}.issubset(types)
    check_both_types(splits.train)
    check_both_types(splits.val)
    check_both_types(splits.test)

    # verify counts roughly match the requested shares
    total_ids = n_samples
    actual_train = len(train_ids)
    actual_val = len(val_ids)
    actual_test = len(test_ids)

    expected_test = int(round(total_ids * TEST_PERCENT))
    expected_val = int(round(total_ids * VAL_PERCENT))
    expected_train = total_ids - expected_val - expected_test

    # allow small rounding artifact tolerance of 1 sample
    assert abs(actual_train - expected_train) <= 1, f"train: got {actual_train}, expected ~{expected_train}"
    assert abs(actual_val - expected_val) <= 1, f"val: got {actual_val}, expected ~{expected_val}"
    assert abs(actual_test - expected_test) <= 1, f"test: got {actual_test}, expected ~{expected_test}"


def test_missing_columns_raises() -> None:
    df = make_balanced_df()
    df = df.drop(columns=[PH2_CLINICAL_DIAGNOSIS_HEADER])
    with pytest.raises(ValueError):
        stratified_split_segmentation_metadata(df, stratify_column=PH2_CLINICAL_DIAGNOSIS_HEADER)


def test_class_with_one_sample_raises() -> None:
    # create classes where one class has only 1 sample
    rows = []
    # class a -> 1 sample
    rows.append({SAMPLEID_HEADER: "s0", DATATYPE_HEADER: DATATYPE_IMAGE, PH2_CLINICAL_DIAGNOSIS_HEADER: "a"})
    rows.append({SAMPLEID_HEADER: "s0", DATATYPE_HEADER: DATATYPE_MASK, PH2_CLINICAL_DIAGNOSIS_HEADER: "a"})
    # class b -> 2 samples
    for i in range(2):
        sid = f"sb{i}"
        rows.append({SAMPLEID_HEADER: sid, DATATYPE_HEADER: DATATYPE_IMAGE, PH2_CLINICAL_DIAGNOSIS_HEADER: "b"})
        rows.append({SAMPLEID_HEADER: sid, DATATYPE_HEADER: DATATYPE_MASK, PH2_CLINICAL_DIAGNOSIS_HEADER: "b"})
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError):
        stratified_split_segmentation_metadata(df, stratify_column=PH2_CLINICAL_DIAGNOSIS_HEADER)


def test_sample_with_duplicate_types_is_excluded() -> None:
    """
    If a sample ID has duplicate image or mask rows (not exactly one of each),
    it should be excluded from the valid sample set and therefore not appear in any split.
    """

    # build a small balanced dataset with enough valid samples
    df = make_balanced_df(n_samples=10, n_classes=2)
    # add a bad sample that has two images and one mask
    bad_rows = [
        {SAMPLEID_HEADER: "bad", DATATYPE_HEADER: DATATYPE_IMAGE, PH2_CLINICAL_DIAGNOSIS_HEADER: "c0"},
        {SAMPLEID_HEADER: "bad", DATATYPE_HEADER: DATATYPE_IMAGE, PH2_CLINICAL_DIAGNOSIS_HEADER: "c0"},
        {SAMPLEID_HEADER: "bad", DATATYPE_HEADER: DATATYPE_MASK, PH2_CLINICAL_DIAGNOSIS_HEADER: "c0"},
    ]
    df = pd.concat([df, pd.DataFrame(bad_rows)], ignore_index=True)

    splits: DataFrameSplits = stratified_split_segmentation_metadata(
        df, stratify_column=PH2_CLINICAL_DIAGNOSIS_HEADER, random_seed=0
    )

    train_ids = set(splits.train[SAMPLEID_HEADER].unique())
    val_ids = set(splits.val[SAMPLEID_HEADER].unique())
    test_ids = set(splits.test[SAMPLEID_HEADER].unique())

    assert "bad" not in train_ids
    assert "bad" not in val_ids
    assert "bad" not in test_ids
