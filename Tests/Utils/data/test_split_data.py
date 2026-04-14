
import pytest
import pandas as pd
from SkiNet.Utils.data.split_data import split_segmentation_metadata, DataFrameSplits, SplitConfig
from SkiNet.Utils.csv_headers import SAMPLEID_HEADER, DATATYPE_HEADER, PH2_CLINICAL_DIAGNOSIS_HEADER, DATATYPE_IMAGE, DATATYPE_MASK

NSAMPLES = 100
TRAIN_PERCENT = 0.7
VAL_PERCENT = 0.15
TEST_PERCENT = 0.15


def make_split_config(stratify_column: str | None = PH2_CLINICAL_DIAGNOSIS_HEADER) -> SplitConfig:
    return SplitConfig(train_size=TRAIN_PERCENT,
                       val_size=VAL_PERCENT,
                       test_size=TEST_PERCENT,
                       stratify_column=stratify_column,
                       random_seed=0)


def make_balanced_df(n_samples: int = NSAMPLES, n_classes: int = 2) -> pd.DataFrame:
    """
    Create a balanced DataFrame with the specified number of samples and classes.
    """
    rows = []
    classes = [f"c{i}" for i in range(n_classes)]
    for i in range(n_samples):
        cid = f"s{i}"
        label = classes[i % n_classes]
        rows.append({SAMPLEID_HEADER: cid, DATATYPE_HEADER: DATATYPE_IMAGE, PH2_CLINICAL_DIAGNOSIS_HEADER: label})
        rows.append({SAMPLEID_HEADER: cid, DATATYPE_HEADER: DATATYPE_MASK, PH2_CLINICAL_DIAGNOSIS_HEADER: label})
    return pd.DataFrame(rows)


def test_splitconfig_post_init_invalid_proportions() -> None:
    """
    SplitConfig should validate that train/val/test proportions sum to 1.0.
    """
    with pytest.raises(ValueError):
        SplitConfig(train_size=0.5, val_size=0.5, test_size=0.2,
                    stratify_column=None, random_seed=0)


def test_splitconfig_post_init_missing_random_seed() -> None:
    """
    SplitConfig should require a random_seed (not None) (stratify is not required)
    """
    with pytest.raises(ValueError):
        SplitConfig(train_size=TRAIN_PERCENT, val_size=VAL_PERCENT, test_size=TEST_PERCENT,
                    stratify_column=None, random_seed=None)


@pytest.mark.parametrize("n_samples,n_classes,stratify_column",
                         [(NSAMPLES, 2, PH2_CLINICAL_DIAGNOSIS_HEADER), (2 * NSAMPLES, 3, PH2_CLINICAL_DIAGNOSIS_HEADER),
                          (NSAMPLES, 2, None)])
def test_normal_split_no_overlap_and_both_types(n_samples: int, n_classes: int, stratify_column: str | None) -> None:
    """
    Test that the stratified split produces non-overlapping splits where each sample ID appears in only one split,
    and that each sample in each split has both image and mask datatypes.
    Also verify that the counts of samples in each split roughly match the requested proportions.
    """
    # Arrange - create a balanced DataFrame with the specified number of samples and classes, and perform the split
    df = make_balanced_df(n_samples=n_samples, n_classes=n_classes)

    # Act - perform the split using the function under test
    splits: DataFrameSplits = split_segmentation_metadata(
        df=df,
        split_config=make_split_config(stratify_column=stratify_column))

    # Assert - verify that the splits are non-overlapping and each sample has both image and mask datatypes,
    #  and that counts roughly match requested proportions
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

    assert abs(actual_train - expected_train) <= 1, f"train: got {actual_train}, expected ~{expected_train}"
    assert abs(actual_val - expected_val) <= 1, f"val: got {actual_val}, expected ~{expected_val}"
    assert abs(actual_test - expected_test) <= 1, f"test: got {actual_test}, expected ~{expected_test}"

    # if stratification is enabled, verify that each split contains at least one sample of each class
    # (more robust than strict equality of class distributions)
    def classes_in_split_by_sample(df_part: pd.DataFrame) -> set:
        # derive classes at the sample-id level (each sample has image+mask rows with same label)
        # we have image and mask for each sample id, take just one of them for counting purposes
        return set(df_part.drop_duplicates(SAMPLEID_HEADER)[PH2_CLINICAL_DIAGNOSIS_HEADER].unique())

    expected_classes = {f"c{i}" for i in range(n_classes)}
    # require at least one sample of every class in each split (more robust than strict equality)
    assert expected_classes.issubset(classes_in_split_by_sample(splits.train))
    assert expected_classes.issubset(classes_in_split_by_sample(splits.val))
    assert expected_classes.issubset(classes_in_split_by_sample(splits.test))

    # check if the actual class proportions in each split are roughly similar to the original proportions
    # (allowing for some noise from discrete sample counts)
    if stratify_column is not None:
        def sample_level_class_proportions(df_part: pd.DataFrame) -> pd.Series:
            # we have image and mask for each sample id, take just one
            sample_df = df_part.drop_duplicates(SAMPLEID_HEADER)
            return sample_df[PH2_CLINICAL_DIAGNOSIS_HEADER].value_counts(normalize=True).sort_index()

        original_props = sample_level_class_proportions(df)
        train_props = sample_level_class_proportions(splits.train)
        val_props = sample_level_class_proportions(splits.val)
        test_props = sample_level_class_proportions(splits.test)

        # allow some rounding noise from discrete sample counts in each split
        tolerance = 0.20

        # so proportions should be roughly similar in each split to the original proportions
        # (not exact because of discrete sample counts, but should be close)
        for cls, expected_prop in original_props.items():
            assert abs(train_props.get(cls, 0.0) - expected_prop) <= tolerance
            assert abs(val_props.get(cls, 0.0) - expected_prop) <= tolerance
            assert abs(test_props.get(cls, 0.0) - expected_prop) <= tolerance


def test_missing_columns_raises() -> None:
    """
    Test that if required columns are missing from the input DataFrame, a ValueError is raised.
    """
    df = make_balanced_df()
    df = df.drop(columns=[PH2_CLINICAL_DIAGNOSIS_HEADER])
    with pytest.raises(ValueError):
        # default split config column is PH2_CLINICAL_DIAGNOSIS_HEADER, so this should raise because it's missing
        split_segmentation_metadata(df=df, split_config=make_split_config())


def test_class_with_one_sample_raises() -> None:
    """
    Test that if a class has only one or two samples, sample, a ValueError is raised because stratified
    splitting cannot be performed (requires at least 3 samples per class to split into train/val/test)
    - may need to adjust for a more realistic splits with small datasets,
    but the key is that if a class has too few samples, it should raise an error when stratification is requested.
    """
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
        split_segmentation_metadata(df=df, split_config=make_split_config())


def test_sample_with_duplicate_types_is_excluded() -> None:
    """
    If a sample ID has duplicate image or mask rows (not exactly one of each),
    it should be excluded from the valid sample set and therefore not appear in any split.
    - maybe some kind of logging or a report about excluded samples would be good in he future
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

    splits: DataFrameSplits = split_segmentation_metadata(
        df=df,
        split_config=make_split_config())

    train_ids = set(splits.train[SAMPLEID_HEADER].unique())
    val_ids = set(splits.val[SAMPLEID_HEADER].unique())
    test_ids = set(splits.test[SAMPLEID_HEADER].unique())

    assert "bad" not in train_ids
    assert "bad" not in val_ids
    assert "bad" not in test_ids


def test_missing_image_or_mask_is_excluded() -> None:
    """
    Test that if a sample ID is missing either an image or a mask, it is excluded from the valid sample set and does not appear in any split.
    """
    # create one good sample and one bad sample missing mask
    rows = [
        {SAMPLEID_HEADER: "good1", DATATYPE_HEADER: DATATYPE_IMAGE, PH2_CLINICAL_DIAGNOSIS_HEADER: "a"},
        {SAMPLEID_HEADER: "good1", DATATYPE_HEADER: DATATYPE_MASK, PH2_CLINICAL_DIAGNOSIS_HEADER: "a"},
        {SAMPLEID_HEADER: "good2", DATATYPE_HEADER: DATATYPE_IMAGE, PH2_CLINICAL_DIAGNOSIS_HEADER: "a"},
        {SAMPLEID_HEADER: "good2", DATATYPE_HEADER: DATATYPE_MASK, PH2_CLINICAL_DIAGNOSIS_HEADER: "a"},
        {SAMPLEID_HEADER: "good3", DATATYPE_HEADER: DATATYPE_IMAGE, PH2_CLINICAL_DIAGNOSIS_HEADER: "a"},
        {SAMPLEID_HEADER: "good3", DATATYPE_HEADER: DATATYPE_MASK, PH2_CLINICAL_DIAGNOSIS_HEADER: "a"},
        {SAMPLEID_HEADER: "bad_no_mask", DATATYPE_HEADER: DATATYPE_IMAGE, PH2_CLINICAL_DIAGNOSIS_HEADER: "a"},
    ]
    df = pd.DataFrame(rows)
    splits = split_segmentation_metadata(df=df, split_config=make_split_config(stratify_column=None))
    all_ids = set(splits.train[SAMPLEID_HEADER]) | set(splits.val[SAMPLEID_HEADER]) | set(splits.test[SAMPLEID_HEADER])
    assert "bad_no_mask" not in all_ids
    assert {"good1", "good2", "good3"}.issubset(all_ids)


def test_reproducible_with_random_seed() -> None:
    """
    Test that using the same random seed produces the same splits (same sample IDs in each split).
    """
    df = make_balanced_df(n_samples=60, n_classes=3)
    cfg = make_split_config()
    cfg = SplitConfig(train_size=cfg.train_size, val_size=cfg.val_size, test_size=cfg.test_size,
                      stratify_column=cfg.stratify_column, random_seed=123)
    s1 = split_segmentation_metadata(df=df, split_config=cfg)
    s2 = split_segmentation_metadata(df=df, split_config=cfg)
    assert set(s1.train[SAMPLEID_HEADER]) == set(s2.train[SAMPLEID_HEADER])
    assert set(s1.val[SAMPLEID_HEADER]) == set(s2.val[SAMPLEID_HEADER])
    assert set(s1.test[SAMPLEID_HEADER]) == set(s2.test[SAMPLEID_HEADER])


def test_different_seed_changes_splits() -> None:
    """
    Test that using different random seeds produces different splits (different sample IDs in at least one split).
    """
    df = make_balanced_df(n_samples=60, n_classes=3)
    cfg1 = make_split_config()
    cfg1 = SplitConfig(train_size=cfg1.train_size, val_size=cfg1.val_size, test_size=cfg1.test_size,
                       stratify_column=cfg1.stratify_column, random_seed=1)
    cfg2 = SplitConfig(train_size=cfg1.train_size, val_size=cfg1.val_size, test_size=cfg1.test_size,
                       stratify_column=cfg1.stratify_column, random_seed=2)
    s1 = split_segmentation_metadata(df=df, split_config=cfg1)
    s2 = split_segmentation_metadata(df=df, split_config=cfg2)
    # Expect at least one split to differ in sample IDs
    assert (set(s1.train[SAMPLEID_HEADER]) != set(s2.train[SAMPLEID_HEADER])
            or set(s1.val[SAMPLEID_HEADER]) != set(s2.val[SAMPLEID_HEADER])
            or set(s1.test[SAMPLEID_HEADER]) != set(s2.test[SAMPLEID_HEADER]))


def test_ambiguous_label_sample_is_excluded() -> None:
    """
    Test that if a sample ID has inconsistent labels across its rows (e.g. different values in the stratify column),
    it is excluded from the valid sample set and does not appear in any split.
    """
    # build a small balanced dataset and add a sample with inconsistent labels
    df = make_balanced_df(n_samples=10, n_classes=2)
    amb = [
        {SAMPLEID_HEADER: "amb", DATATYPE_HEADER: DATATYPE_IMAGE, PH2_CLINICAL_DIAGNOSIS_HEADER: "x"},
        {SAMPLEID_HEADER: "amb", DATATYPE_HEADER: DATATYPE_MASK, PH2_CLINICAL_DIAGNOSIS_HEADER: "y"},
    ]
    df = pd.concat([df, pd.DataFrame(amb)], ignore_index=True)
    splits = split_segmentation_metadata(df=df, split_config=make_split_config())
    all_ids = set(splits.train[SAMPLEID_HEADER]) | set(splits.val[SAMPLEID_HEADER]) | set(splits.test[SAMPLEID_HEADER])
    assert "amb" not in all_ids
