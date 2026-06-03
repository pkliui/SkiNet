from enum import Enum, StrEnum, unique
import re


@unique
class DatasetKey(Enum):
    """
    Enum for dataset keys used to identify datasets in Azure.

    The VALUE of each enum member is a unique string identifier for data handled in this code and we call it `dataset_name` (for YAML/config/Azure).
    The enum member itself (e.g. DatasetKey.PH2) is `dataset_key` (for code logic).
    The short string (e.g., 'PH2') is `dataset_key_str` (for CLI, etc).
    The `dataset_name` must match one of the keys in PATH_ON_DATASTORE of your Azure YAML config.
    It is used to retrieve the correct path on the Azure datastore for that dataset.
    """
    PH2 = "PH2_DATASET"
    ISIC2017 = "ISIC2017_DATASET"


@unique
class ModelKey(Enum):
    """
    Enum for model keys used to identify models.

    The VALUE of each enum member is a unique string identifier for a model handled in this code and we call it `model_name` (for YAML/config/Azure).
    The enum member itself (e.g. ModelKey.UNET2D) is `model_key` (for code logic).
    The short string (e.g., 'UNET2D') is `model_key_str` (for CLI, etc).
    The `model_name` is the string value that should appear under the "MODEL" key in your YAML config (e.g., Azure settings).
    """
    UNET2D = "UNET2D_MODEL"


@unique
class ExperimentType(Enum):
    """
    Enum for experiment types.

    For example, they are used in dataset factories for identification of experiment type and for retrieving the correct dataset factory.
    """
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"


@unique
class LossFunctionKey(Enum):
    """
    Enum for loss function keys used to identify loss functions.

    For example, they are used in Lightning model.
    """
    BCE = "bce"
    DICE = "dice"
    BCE_DICE = "bce_dice"


@unique
class HyperparamKey(str, Enum):
    """
    Enum for hyperparameter keys used in the Optuna sweep search space.

    Inherits from str so members can be used directly as dict keys against
    plain-str-keyed dicts (e.g. search_space[HyperparamKey.LR]).
    Adding a new hyperparameter here automatically makes validate_search_space
    require it, preventing silent mismatches between the search space and objective.
    """
    LR = "lr"
    WEIGHT_DECAY = "weight_decay"
    BATCH_SIZE = "batch_size"
    NUM_WORKERS = "num_workers"
    PREFETCH_FACTOR = "prefetch_factor"
    SCHEDULER_TYPE = "scheduler_type"


@unique
class MetricsKey(str, Enum):
    """
    Enum for metric keys passed to external frameworks (Lightning, MLflow, Optuna).

    Inherits from str so each member IS a str at the Python object level. This means
    dict lookups against trainer.callback_metrics (which has plain str keys) succeed
    without calling .value explicitly. With a plain Enum, the member's __hash__ and
    __eq__ differ from the equivalent str, so the lookup silently misses even though
    the .value is correct.
    """
    VAL_BEST_DICE_AT_THRESHOLD = "val_best_dice_at_threshold"

    @classmethod
    def default_monitor(cls) -> "MetricsKey":
        """
        Default used in Optuna and EarlyStopping.
        """
        return cls.VAL_BEST_DICE_AT_THRESHOLD


# =========================================================
# RESIDUAL MODE VOCABULARIES
# Values must stay in sync with the registry keys in encoder2d.py / merge2d_block.py.
# =========================================================

@unique
class EncoderResidualMode(StrEnum):
    CLASSICAL = "classical"
    LOCAL_REFINEMENT = "local_refinement"
    HE2 = "he2"
    SE = "se"


@unique
class MergeResidualMode(StrEnum):
    CLASSICAL = "classical"
    LOCAL_REFINEMENT = "local_refinement"
    HE1 = "he1"
    HE2 = "he2"
    ATTENTION_GATE = "attention_gate"


# =========================================================
# NETWORK BLOCK PARSING
# =========================================================

class NetworkBlockKey(Enum):
    """
    Class to hold the prefixes and regex pattern for parsing encoder/merge block keys from experiment names.

    Experiment names encode the encoder/merge modes (e.g. "ablation_model_unet2d-ph2_enc-he2_merge-attention_gate")
    and we need to parse these back out to know which modes are active for a given run.


    Example usage:
    ```
    for enc, merge in combos:
        # Each (enc, merge) combination gets its own MLflow experiment so runs
        # are unambiguous in the UI even if the study is interrupted and resumed.
        experiment_name = (f"{args.name_prefix}"
                           f"_{NetworkBlockKey.ENC_PREFIX.value}{enc}"
                           f"_{NetworkBlockKey.MERGE_PREFIX.value}{merge}")
    ```
    Which for enc and merge modes of "he2" and "attention_gate", name_prefix  of "model"
    would produce an experiment name like:
    "model_enc-he2_merge-attention_gate"
    """
    ENC_PREFIX = "enc-"
    MERGE_PREFIX = "merge-"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def arch_pattern(cls) -> re.Pattern:
        enc = re.escape(cls.ENC_PREFIX.value)
        mrg = re.escape(cls.MERGE_PREFIX.value)

        # value stops BEFORE next _enc or _merge or end
        value = r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*"

        return re.compile(
            rf"(?:^|_)"
            rf"{enc}({value})"
            rf"(?=_(?:merge-|$))"
            rf".*?"
            rf"(?:^|_)"
            rf"{mrg}({value})"
            rf"(?=_|$)"
        )
