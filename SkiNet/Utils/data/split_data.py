from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

from SkiNet.Utils.csv_headers import SAMPLEID_HEADER, DATATYPE_HEADER, DATATYPE_IMAGE, DATATYPE_MASK

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitConfig:
    """
    Container for the stratified split of the DataFrame specifying the proportions for train/val/test splits,
    an optional column name for stratification, and a random seed for reproducibility.
    """
    train_size: float
    val_size: float
    test_size: float
    stratify_column: str | None
    random_seed: Optional[int]

    def __post_init__(self) -> None:
        total = self.train_size + self.val_size + self.test_size
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"Split proportions must sum to 1.0, got {total}")
        if not (0.0 < self.train_size < 1.0 and 0.0 <= self.val_size < 1.0 and 0.0 <= self.test_size < 1.0):
            raise ValueError("train/val/test sizes must be in (0,1) and sum to 1.0")
        if self.random_seed is None:
            raise ValueError("random_seed must be provided")


@dataclass
class DataFrameSplits:
    """
    Container to keep  train, validation, and test splits of a DataFrame after performing splitting based on sample IDs.
    """
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _validate_inputs(df: pd.DataFrame, stratify_column: str | None) -> None:
    """
    Validate required columns exist. If use_stratify is False, the stratify_column is not required.
    """
    required_cols = {SAMPLEID_HEADER, DATATYPE_HEADER}
    if stratify_column is not None:
        required_cols.add(stratify_column)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns required for stratified split in the original dataframe: {missing}")


def _collect_valid_samples(df: pd.DataFrame,
                           stratify_column: str | None) -> pd.DataFrame:
    """
    Collects valid sample IDs from the input DataFrame.
    Require there is exactly one image and one mask per sample ID,
    and a single, non-empty label for the stratify column if stratification is enabled.
    """
    # Just a temporary list to keep track of valid sample IDs and their corresponding stratify labels for the splitting process
    valid_sample_rows: List[Dict[str, Any]] = []
    # Group by sample ID
    for sample_id, rows_with_sampleid in df.groupby(SAMPLEID_HEADER):
        # Ensure both image and mask are present for this sample ID by checking the datatype column
        datatypes = set(rows_with_sampleid[DATATYPE_HEADER].astype(str))
        if not {DATATYPE_IMAGE, DATATYPE_MASK}.issubset(datatypes):
            logger.warning("Skipping sample %s because of a missing image or mask", sample_id)
            continue

        # Enforce exactly one image and one mask (avoid duplicates / extra rows per sample id)
        counts = rows_with_sampleid[DATATYPE_HEADER].astype(str).value_counts()
        if counts.get(DATATYPE_IMAGE, 0) != 1 or counts.get(DATATYPE_MASK, 0) != 1:
            logger.warning("Skipping sample %s because it has %d images and %d masks", sample_id,
                           counts.get(DATATYPE_IMAGE, 0), counts.get(DATATYPE_MASK, 0))
            continue

        sample_row: Dict[str, Any] = {SAMPLEID_HEADER: sample_id}

        if stratify_column is not None:
            # Within each rows_with_sampleid, extract the UNIQUE, non-empty labels for the stratify column.
            # This excludes rows with ambiguous or missing labels from stratified splitting.
            labels = rows_with_sampleid[stratify_column].dropna().astype(str).str.strip().unique()
            labels = [x for x in labels if x != ""]
            if len(labels) != 1:
                continue
            sample_row[stratify_column] = labels[0]

        valid_sample_rows.append(sample_row)

    validated_sampleids_df = pd.DataFrame(valid_sample_rows)
    return validated_sampleids_df


def _perform_id_splits(validated_sampleids_df: pd.DataFrame,
                       stratify_column: str | None,
                       train_size: float,
                       val_size: float,
                       test_size: float,
                       random_seed: int | None) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits the valid sample IDs into train, validation, and test sets while ensuring that
    each sample ID is only in one split and that the splits are stratified according to the specified column.
    """
    if len(validated_sampleids_df) < 3:
        raise ValueError("Need at least 3 valid sample IDs for train/val/test splitting.")

    stratify_values = None
    if stratify_column is not None:
        class_counts = validated_sampleids_df[stratify_column].value_counts()
        if (class_counts < 3).any():
            raise ValueError(
                f"Each class in {stratify_column} needs at least 3 samples for stratified splitting. "
                f"Counts:\n{class_counts.to_string()}"
            )
        stratify_values = validated_sampleids_df[stratify_column]

    # Get the IDs for test and train_val splits first, stratifying by the specified column
    train_val_ids, test_ids = train_test_split(
        validated_sampleids_df[SAMPLEID_HEADER],
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
        stratify=stratify_values,
    )

    # Keep only the rows corresponding to the train_val_ids for the next split,
    # and recalculate the stratify fraction for the validation set w.r.t the train_val set
    train_val_df = validated_sampleids_df[validated_sampleids_df[SAMPLEID_HEADER].isin(train_val_ids)].copy()
    val_fraction_of_trainval = val_size / (train_size + val_size)

    # Now split the train_val_ids into train and val, stratifying by the same column
    train_val_stratify = train_val_df[stratify_column] if stratify_column is not None else None
    train_ids, val_ids = train_test_split(
        train_val_df[SAMPLEID_HEADER],
        test_size=val_fraction_of_trainval,
        random_state=random_seed,
        shuffle=True,
        stratify=train_val_stratify,
    )
    return train_ids, val_ids, test_ids


def split_segmentation_metadata(df: pd.DataFrame,
                                split_config: SplitConfig) -> DataFrameSplits:
    """
    Returns stratified train/val/test splits of the input DataFrame based on sample IDs,
    ensuring that each sample ID (which should correspond to a unique image/mask pair) is only in one split and that
    the splits are stratified according to the specified column.

    :param df: The input DataFrame containing metadata about the segmentation dataset, including sample IDs, datatypes (image/mask), and stratify labels.
    :param split_config: Split configuration. If split_config.stratify_column is None, a non-stratified split is used.
        Obtained from the dataset config's get_split_config() method as follows:
        ```python
        main_config: ExperimentConfig = load_config_from_yaml(cfg_path)
        split_config: SplitConfig = main_config.dataconfig.get_split_config()
        ```

    :return: A DataFrameSplits object containing the train, validation, and test splits of the original DataFrame.
    """
    _validate_inputs(df, split_config.stratify_column)
    validated_sampleids_df = _collect_valid_samples(df, split_config.stratify_column)
    train_ids, val_ids, test_ids = _perform_id_splits(validated_sampleids_df,
                                                      split_config.stratify_column,
                                                      split_config.train_size,
                                                      split_config.val_size,
                                                      split_config.test_size,
                                                      split_config.random_seed)

    # Use the original df to get the full rows for the train, val, and test splits based on the sample IDs we just determined
    train_df: pd.DataFrame = df[df[SAMPLEID_HEADER].isin(train_ids)].copy()
    val_df: pd.DataFrame = df[df[SAMPLEID_HEADER].isin(val_ids)].copy()
    test_df: pd.DataFrame = df[df[SAMPLEID_HEADER].isin(test_ids)].copy()

    return DataFrameSplits(train=train_df, val=val_df, test=test_df)
