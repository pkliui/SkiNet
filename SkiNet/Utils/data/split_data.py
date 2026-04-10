from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from SkiNet.Utils.csv_headers import SAMPLEID_HEADER, DATATYPE_HEADER, DATATYPE_IMAGE, DATATYPE_MASK
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class DataFrameSplits:
    """
    Keeps track of the train, validation, and test splits of a DataFrame after performing stratified splitting based on sample IDs.
    """
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _validate_inputs(df: pd.DataFrame, sampleid_column: str, datatype_column: str, stratify_column: str) -> None:
    """
    Validates that the input DataFrame contains the necessary columns
    for performing a stratified split based on sample IDs and datatypes.
    """
    # Check if any of the required columns are missing from the DataFrame
    required_cols = {sampleid_column, datatype_column, stratify_column}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns required for stratified split in the original dataframe: {missing}")


def _collect_valid_samples(df: pd.DataFrame, sampleid_column: str, datatype_column: str, stratify_column: str) -> pd.DataFrame:
    """
    Collects valid sample IDs from the input DataFrame that have exactly one image and one mask,
    and a single, non-empty label for the stratify column.
    """
    # Just a temporary list to keep track of valid sample IDs and their corresponding stratify labels for the splitting process
    valid_sample_rows: List[Dict[str, Any]] = []
    # Group by sample ID
    for sample_id, rows_with_sampleid in df.groupby(sampleid_column):
        # Ensure both image and mask are present for this sample ID by checking the datatype column
        datatypes = set(rows_with_sampleid[datatype_column].astype(str))
        logger.warning("Skipping sample %s because of a missing image or mask", sample_id)
        if not {DATATYPE_IMAGE, DATATYPE_MASK}.issubset(datatypes):
            continue

        # Enforce exactly one image and one mask (avoid duplicates / extra rows per sample id)
        counts = rows_with_sampleid[datatype_column].astype(str).value_counts()
        if counts.get(DATATYPE_IMAGE, 0) != 1 or counts.get(DATATYPE_MASK, 0) != 1:
            logger.warning("Skipping sample %s because it has %d images and %d masks", sample_id,
                           counts.get(DATATYPE_IMAGE, 0), counts.get(DATATYPE_MASK, 0))
            continue

        # Within each rows_with_sampleid, extract the UNIQUE, non-empty labels for the stratify column
        # Means rows with ambigouous or missing labels will be excluded
        labels = rows_with_sampleid[stratify_column].dropna().astype(str).str.strip().unique()
        labels = [x for x in labels if x != ""]
        if len(labels) != 1:
            continue

        valid_sample_rows.append({
            sampleid_column: sample_id,
            stratify_column: labels[0],
        })

    validated_sampleids_df = pd.DataFrame(valid_sample_rows)
    return validated_sampleids_df


def _perform_id_splits(validated_sampleids_df: pd.DataFrame,
                       sampleid_column: str,
                       stratify_column: str,
                       train_size: float,
                       val_size: float,
                       test_size: float,
                       random_seed: int) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits the valid sample IDs into train, validation, and test sets while ensuring that
    each sample ID is only in one split and that the splits are stratified according to the specified column.
    """
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split proportions must sum to 1.0, got {total}")

    if len(validated_sampleids_df) < 3:
        raise ValueError("Need at least 3 valid sample IDs for train/val/test splitting.")

    class_counts = validated_sampleids_df[stratify_column].value_counts()
    if (class_counts < 2).any():
        raise ValueError(
            f"Each class in {stratify_column} needs at least 2 samples for stratified splitting. "
            f"Counts:\n{class_counts.to_string()}"
        )

    # Get the IDs for test and train_val splits first, stratifying by the specified column
    train_val_ids, test_ids = train_test_split(
        validated_sampleids_df[sampleid_column],
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
        stratify=validated_sampleids_df[stratify_column],
    )

    # Keep only the rows corresponding to the train_val_ids for the next split,
    # and recalculate the stratify fraction for the validation set w.r.t the train_val set
    train_val_df = validated_sampleids_df[validated_sampleids_df[sampleid_column].isin(train_val_ids)].copy()
    val_fraction_of_trainval = val_size / (train_size + val_size)

    # Now split the train_val_ids into train and val, stratifying by the same column
    train_ids, val_ids = train_test_split(
        train_val_df[sampleid_column],
        test_size=val_fraction_of_trainval,
        random_state=random_seed,
        shuffle=True,
        stratify=train_val_df[stratify_column],
    )
    return train_ids, val_ids, test_ids


def stratified_split_segmentation_metadata(df: pd.DataFrame,
                                           stratify_column: str,
                                           sampleid_column: str = SAMPLEID_HEADER,
                                           datatype_column: str = DATATYPE_HEADER,
                                           train_size: float = 0.7,
                                           val_size: float = 0.1,
                                           test_size: float = 0.2,
                                           random_seed: int = 42) -> DataFrameSplits:
    """
    Returns stratified train/val/test splits of the input DataFrame based on sample IDs,
    ensuring that each sample ID (which should correspond to a unique image/mask pair) is only in one split and that
    the splits are stratified according to the specified column.

    :param df: The input DataFrame containing metadata about the segmentation dataset, including sample IDs, datatypes (image/mask), and stratify labels.
    :param stratify_column: The name of the column in the DataFrame to use for stratification (e.g., a class label).
    :param sampleid_column: The name of the column in the DataFrame that contains sample IDs.
    :param datatype_column: The name of the column in the DataFrame that indicates whether a row corresponds to an image or a mask.
    :param train_size: The proportion of the dataset to include in the train split (default 0.7).
    :param val_size: The proportion of the dataset to include in the validation split (default 0.1).
    :param test_size: The proportion of the dataset to include in the test split (default 0.2).
    :param random_seed: The random seed to use for reproducibility of the splits (default 42).

    :return: A DataFrameSplits object containing the train, validation, and test splits of the original DataFrame.
    """
    _validate_inputs(df, sampleid_column, datatype_column, stratify_column)
    validated_sampleids_df = _collect_valid_samples(df, sampleid_column, datatype_column, stratify_column)
    train_ids, val_ids, test_ids = _perform_id_splits(validated_sampleids_df,
                                                      sampleid_column,
                                                      stratify_column,
                                                      train_size,
                                                      val_size,
                                                      test_size,
                                                      random_seed)

    # Use the original df to get the full rows for the train, val, and test splits based on the sample IDs we just determined
    train_df = df[df[sampleid_column].isin(train_ids)].copy()
    val_df = df[df[sampleid_column].isin(val_ids)].copy()
    test_df = df[df[sampleid_column].isin(test_ids)].copy()

    return DataFrameSplits(train=train_df, val=val_df, test=test_df)
