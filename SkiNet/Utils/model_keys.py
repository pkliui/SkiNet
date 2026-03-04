from enum import Enum, unique


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
