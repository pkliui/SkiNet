import argparse
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from SkiNet.ML.datasets.preprocessing.base_csv_builder import AzureCSVBuilder, BaseCSVBuilder, LocalCSVBuilder
from SkiNet.Utils.csv_headers import (ISIC2017_IMAGE_ID_HEADER, ISIC2017_MELANOMA_HEADER,
                                      ISIC2017_PREDEFINED_SPLIT_HEADER, ISIC2017_SEBORRHEIC_KERATOSIS_HEADER,
                                      SAMPLEID_HEADER)
from SkiNet.Utils.experiment_keys import DatasetKey
from SkiNet.Utils.project_paths import (ISIC2017_CSV_NAME, ISIC2017_IMAGE_PATTERN_AZURE,
                                        ISIC2017_IMAGE_PATTERN_LOCAL, ISIC2017_MASK_PATTERN_AZURE,
                                        ISIC2017_MASK_PATTERN_LOCAL, ISIC2017_TEST_GT_CSV_NAME,
                                        ISIC2017_TRAIN_GT_CSV_NAME, ISIC2017_VAL_GT_CSV_NAME)

_SPLIT_DIR_MAP = {"Training": "train",
                  "Validation": "val",
                  "Test": "test"}


class ISIC2017BaseCSVBuilder(BaseCSVBuilder, ABC):
    """
    Base class to build CSV metadata for the ISIC 2017 dataset from the file structure and
    the provided ground-truth diagnosis CSV files.

    ISIC 2017 dataset challenge details and file structure are described in the original challenge paper:
    Codella N, Gutman D, Celebi ME, Helba B, Ettlin MA, Marchetti MA, Dusza S, Kalloo A, Liopyris K,
    Mishra N, Kittler H, Halpern A. "Skin Lesion Analysis Toward Melanoma Detection: ISIC 2017 Challenge."
    arXiv:1710.05006 [cs.CV], 2017.

    The expected directory structure (as per original ISIC 2017 file layout):
    * root_dir/
        * ISIC-2017_Training_Data/
            * ISIC_0000000.jpg
            * ISIC_0000001.jpg
            * ...
        * ISIC-2017_Training_Part1_GroundTruth/
            * ISIC_0000000_segmentation.png
            * ISIC_0000001_segmentation.png
            * ...
        * ISIC-2017_Training_Part3_GroundTruth.csv
        * ISIC-2017_Validation_Data/
            * ...
        * ISIC-2017_Validation_Part1_GroundTruth/
            * ...
        * ISIC-2017_Validation_Part3_GroundTruth.csv
        * ISIC-2017_Test_v2_Data/
            * ...
        * ISIC-2017_Test_v2_Part1_GroundTruth/
            * ...
        * ISIC-2017_Test_v2_Part3_GroundTruth.csv
    """

    @property
    def output_csv_name(self) -> str:
        """ The name of the output CSV file for the ISIC2017 dataset. """
        return ISIC2017_CSV_NAME

    def sampleid_func(self, path_str: str) -> str:
        """
        Extract the sample ID from an image or mask path.

        Image:  .../ISIC-2017_Training_Data/ISIC_0000000.jpg       -> "ISIC_0000000"
        Mask:   .../ISIC-2017_Training_Part1_GroundTruth/ISIC_0000000_segmentation.png -> "ISIC_0000000"

        :param path_str: The image (or mask) path as a string.
            Can be either an absolute or relative path, local or Azure path.
        :return: The sample ID extracted from the image (or mask) path.
        """
        stem = Path(path_str).stem
        return stem.replace("_segmentation", "")

    def predefined_split_func(self, path_str: str) -> str:
        """
        Determine the predefined split (train/val/test) from an image or mask path
        by inspecting the parent directory name.
        """
        parent = Path(path_str).parent.name
        for key, split in _SPLIT_DIR_MAP.items():
            if key in parent:
                return split
        return "unknown"

    @abstractmethod
    def load_diagnosis_csv(self, csv_name: str) -> pd.DataFrame:
        """
        Load one of the ISIC 2017 ground-truth diagnosis CSVs.
        Implemented differently for local and Azure environments.
        """
        pass

    def load_all_diagnosis_data(self) -> pd.DataFrame:
        """
        Load and concatenate the three ISIC 2017 ground-truth diagnosis CSVs
        (train, val, test) into one DataFrame.
        """
        dfs = [self.load_diagnosis_csv(name) for name in
               (ISIC2017_TRAIN_GT_CSV_NAME, ISIC2017_VAL_GT_CSV_NAME, ISIC2017_TEST_GT_CSV_NAME)]
        return pd.concat(dfs, ignore_index=True)

    def create_merged_isic2017_metadata(self) -> pd.DataFrame:
        """
        Build the full ISIC 2017 metadata DataFrame by:
        1. Creating the basic (sampleid, datapath, datatype) metadata from file paths.
        2. Adding a predefined_split column derived from the directory structure.
        3. Merging with diagnosis labels (melanoma, seborrheic_keratosis).
        """
        basic_df = self.create_basic_metadata()

        # Add predefined_split from each row's datapath
        basic_df[ISIC2017_PREDEFINED_SPLIT_HEADER] = basic_df["datapath"].apply(self.predefined_split_func)

        diagnosis_df = self.load_all_diagnosis_data()
        # Rename image_id to sampleid for the join
        diagnosis_df = diagnosis_df.rename(columns={ISIC2017_IMAGE_ID_HEADER: SAMPLEID_HEADER})

        merged = basic_df.merge(diagnosis_df[[SAMPLEID_HEADER, ISIC2017_MELANOMA_HEADER, ISIC2017_SEBORRHEIC_KERATOSIS_HEADER]],
                                on=SAMPLEID_HEADER,
                                how="left")
        return merged


class ISIC2017LocalCSVBuilder(ISIC2017BaseCSVBuilder, LocalCSVBuilder):
    """
    Builder for creating CSV metadata for the ISIC 2017 dataset in a local environment.

    Example call:
        builder = ISIC2017LocalCSVBuilder(arg)
        builder.create_metadata_csv()
    """

    def __init__(self, arg: argparse.Namespace) -> None:
        ISIC2017BaseCSVBuilder.__init__(self)
        LocalCSVBuilder.__init__(self, arg)

    @property
    def image_pattern(self) -> str:
        return ISIC2017_IMAGE_PATTERN_LOCAL

    @property
    def mask_pattern(self) -> str:
        return ISIC2017_MASK_PATTERN_LOCAL

    def load_diagnosis_csv(self, csv_name: str) -> pd.DataFrame:
        return pd.read_csv(Path(self.data_root) / csv_name)

    def create_metadata_csv(self) -> None:
        self.save_dataframe_to_csv(df=self.create_merged_isic2017_metadata(),
                                   output_csv_path=str(Path(self.data_root) / self.output_csv_name))


class ISIC2017AzureCSVBuilder(ISIC2017BaseCSVBuilder, AzureCSVBuilder):
    """
    Builder for creating CSV metadata for the ISIC 2017 dataset in an Azure environment.

    Example call:
        builder = ISIC2017AzureCSVBuilder()
        builder.create_metadata_csv()
    """

    def __init__(self) -> None:
        ISIC2017BaseCSVBuilder.__init__(self)
        AzureCSVBuilder.__init__(self, DatasetKey.ISIC2017.value)

    @property
    def image_pattern(self) -> str:
        return ISIC2017_IMAGE_PATTERN_AZURE

    @property
    def mask_pattern(self) -> str:
        return ISIC2017_MASK_PATTERN_AZURE

    def load_diagnosis_csv(self, csv_name: str) -> pd.DataFrame:
        csv_path = f"{self.data_root_on_azure}/{csv_name}"
        with self.fs.open(csv_path, "r") as f:
            return pd.read_csv(f)

    def create_metadata_csv(self) -> None:
        self.save_dataframe_and_upload_csv(df=self.create_merged_isic2017_metadata())
