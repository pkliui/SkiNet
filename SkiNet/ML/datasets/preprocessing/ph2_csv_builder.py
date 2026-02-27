import argparse
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from SkiNet.ML.configs.datasets.dataset_keys import DatasetKey
from SkiNet.ML.datasets.preprocessing.base_csv_builder import AzureCSVBuilder, BaseCSVBuilder, LocalCSVBuilder
from SkiNet.Utils.csv_headers import PH2_COLORS_HEADER, PH2_COLORS_LIST_HEADER, PH2_NAME_HEADER, SAMPLEID_HEADER
from SkiNet.Utils.project_paths import (PH2_CSV_NAME, PH2_IMAGE_PATTERN_AZURE, PH2_IMAGE_PATTERN_LOCAL,
                                        PH2_MASK_PATTERN_AZURE, PH2_MASK_PATTERN_LOCAL, PH2_TXT_NAME)


class PH2BaseCSVBuilder(BaseCSVBuilder, ABC):
    """
    Base class to build CSV metadata for the PH2 dataset from the file structure and an external TXT file containing additional metadata.

    PH2 dataset copyright: Teresa Mendonça, Pedro M. Ferreira, Jorge Marques, Andre R. S. Marcal, Jorge Rozeira.
    PH2 - A dermoscopic image database for research and benchmarking,
    35th International Conference of the IEEE Engineering in Medicine and Biology Society, July 3-7, 2013, Osaka, Japan.

    The file structure is expected to be as following (as per original PH2 file structure):
    * root_dir
        * sample1
            * sample1_Dermoscopic_Image
                * sample1.bmp
            * sample1_lesion
            *    sample1_lesion.bmp
        * sample2
            * sample2_Dermoscopic_Image
                * sample1.bmp
            * sample2_lesion
                * sample2_lesion.bmp

    Note that only ONE image per folder is expected,
    e.g. one image in sample1_Dermoscopic_Image folder, one mask in sample1_lesion folder
    """

    @property
    def external_txt_name(self) -> str:
        """The name of the external TXT file containing additional metadata for the PH2 dataset."""
        return PH2_TXT_NAME

    @property
    def output_csv_name(self) -> str:
        """ The name of the output CSV file for the PH2 dataset. """
        return PH2_CSV_NAME

    @abstractmethod
    def parse_external_txt(self) -> pd.DataFrame:
        """
        Parse the PH2-specific external TXT file into a DataFrame.
        Expected to be implemented differently for local and Azure environments due to differences in file access methods.
        """
        pass

    def sampleid_func(self, path_str: str) -> str:
        """
        Extract the sample ID from the image (or mask) path in PH2 dataset.

        :param path_str: The image (or mask) path as a string.
            Can be either an absolute or relative path, local or Azure path.
        :return: The sample ID extracted from the image (or mask) path.
        """
        return str(Path(path_str).parent.parent.name)

    def create_merged_ph2_metadata(self) -> pd.DataFrame:
        """
        Create a merged DataFrame for the PH2 dataset.
        """
        return self.merge_ph2_data(basic_df=self.create_basic_metadata(),
                                   external_df=self.parse_external_txt())

    def merge_ph2_data(self, basic_df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the PH2 dataset's basic metadata (derived from the file structure) with the additional metadata from the external TXT file

        :param basic_df: DataFrame containing the basic metadata with columns [sampleid, datapath, datatype].
        :param external_df: DataFrame containing the external metadata to merge with the basic metadata.
        :return: A merged DataFrame containing the combined metadata from both sources, with the 'Colors' column processed into a list of integers.
        """
        # Ensure that samples from the basic_df are not lost during the merge, even if they don't have a corresponding entry in the external_df
        df_merged = basic_df.merge(external_df, left_on=SAMPLEID_HEADER, right_on=PH2_NAME_HEADER, how="left")

        # Process the 'Colors' column from the TXT file into a list of integers, and drop the original 'Colors' column
        if PH2_COLORS_HEADER in df_merged.columns:
            df_merged[PH2_COLORS_LIST_HEADER] = df_merged[PH2_COLORS_HEADER].apply(lambda x: [int(i) for i in str(x).split()] if pd.notnull(x) else [])
            df_merged = df_merged.drop(columns=[PH2_COLORS_HEADER])

        return df_merged

    @staticmethod
    def _parse_txt_lines(lines: list[str]) -> pd.DataFrame:
        """
        Parse the lines from the external PH2 TXT file into a DataFrame.

        :param lines: A list of strings representing the lines from the TXT file.
        :return: A DataFrame containing the parsed input from the TXT file.
        """
        FEAT_COL_IDX: int = 3  # The index of the features column in the TXT file where the features are separated by '|'

        def _split_lines(line: str) -> list[str]:
            """
            Split a line from the TXT file into its components.
            The line is expected to have a structure where the first few columns are separated by '||',
            and the features column (at index FEAT_COL_IDX) contains multiple features separated by '|'.

            :param line: A line from the TXT file.
            :return: A list of components extracted from the line.
            """
            # Remove trailing and leading pipes and then split on double pipes
            line_parts = [ll.strip() for ll in line.strip('|').split('||')]
            # The features part needs to be split further
            features = [f.strip() for f in line_parts[FEAT_COL_IDX].split('|')]
            # Processed line with features split into separate columns
            processed_line = line_parts[:FEAT_COL_IDX] + features + line_parts[FEAT_COL_IDX + 1:]
            assert len(processed_line) == len(line_parts) - 1 + len(features), "Line length mismatch after splitting features"
            return processed_line

        header = _split_lines(lines[0])
        data = [_split_lines(line) for line in lines[1:]]
        return pd.DataFrame(data, columns=header)


class PH2LocalCSVBuilder(PH2BaseCSVBuilder, LocalCSVBuilder):
    """
    Builder for creating CSV metadata for the PH2 dataset in a local environment.

    Example call:
    builder = PH2LocalCSVBuilder(arg)
    builder.create_metadata_csv()
    """

    def __init__(self, arg: argparse.Namespace) -> None:
        """
        :param arg: Command-line arguments, containing the root path to data on a local file system.
        """
        PH2BaseCSVBuilder.__init__(self)
        LocalCSVBuilder.__init__(self, arg)

    @property
    def external_txt_path(self) -> str:
        """Path to the external TXT file containing additional metadata for the PH2 dataset, located on the local file system."""
        return f"{self.data_root}/{self.external_txt_name}"

    @property
    def image_pattern(self) -> str:
        """Glob pattern to match image files in the PH2 dataset (local)."""
        return PH2_IMAGE_PATTERN_LOCAL

    @property
    def mask_pattern(self) -> str:
        """Glob pattern to match mask files in the PH2 dataset (local)."""
        return PH2_MASK_PATTERN_LOCAL

    def parse_external_txt(self) -> pd.DataFrame:
        """
        Parse self.external_txt_path from a local file system.
        The TXT file has a specific format where the first line contains headers,
        and subsequent lines contain data. Only lines starting with '||' are considered (header and data lines).

        :return: A DataFrame containing the parsed PH2 dataset metadata.
        """
        # Only keep lines that start with '||', i.e. the header and data lines
        with open(self.external_txt_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip().startswith('||')]
        if not lines:
            raise ValueError(f"External TXT file '{self.external_txt_path}' is empty or contains no valid lines starting with '||'.")
        return self._parse_txt_lines(lines)

    def create_metadata_csv(self) -> None:
        """
        Create a metadata CSV file for the dataset on a local file system.
        """
        self.save_dataframe_to_csv(df=self.create_merged_ph2_metadata(),
                                   output_csv_path=str(Path(self.data_root) / self.output_csv_name))


class PH2AzureCSVBuilder(PH2BaseCSVBuilder, AzureCSVBuilder):
    """
    Builder for creating CSV metadata for the PH2 dataset in an Azure environment.

    Example call:
    builder = PH2AzureCSVBuilder()
    builder.create_metadata_csv()
    """

    def __init__(self) -> None:
        PH2BaseCSVBuilder.__init__(self)
        AzureCSVBuilder.__init__(self, DatasetKey.PH2.value)

    @property
    def external_txt_path(self) -> str:
        """Path to the external TXT file containing additional metadata for the PH2 dataset, located on Azure Blob Storage."""
        return f"{self.data_root_on_azure}/{self.external_txt_name}"

    @property
    def image_pattern(self) -> str:
        """Glob pattern to match image files in the PH2 dataset on Azure."""
        return PH2_IMAGE_PATTERN_AZURE

    @property
    def mask_pattern(self) -> str:
        """Glob pattern to match mask files in the PH2 dataset on Azure."""
        return PH2_MASK_PATTERN_AZURE

    def parse_external_txt(self) -> pd.DataFrame:
        """
        Parse  self.external_txt_path from Azure Blob Storage using the AzureMachineLearningFileSystem.
        The TXT file has a specific format where the first line contains headers,
        and subsequent lines contain data. Only lines starting with '||' are considered (header and data lines).

        :return: A DataFrame containing the parsed PH2 dataset metadata.
        """
        # Only keep lines that start with '||', i.e. the header and data lines
        with self.fs.open(self.external_txt_path, 'r') as f:
            lines = [line.decode('utf-8').strip() for line in f if line.decode('utf-8').strip().startswith('||')]
            return self._parse_txt_lines(lines)

    def create_metadata_csv(self) -> None:
        """
        Create a metadata CSV file for the dataset on Azure
        """
        self.save_dataframe_and_upload_csv(df=self.create_merged_ph2_metadata())
