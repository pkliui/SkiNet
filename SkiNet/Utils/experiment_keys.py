from enum import Enum, unique


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
class MetricsKey(Enum):
    """
    Enum for metrics
    """
    VAL_BEST_DICE_AT_THRESHOLD = "val_best_dice_at_threshold"

    @classmethod
    def default_monitor(cls) -> "MetricsKey":
        """
        Default used in Optuna and EarlyStopping.
        """
        return cls.VAL_BEST_DICE_AT_THRESHOLD
