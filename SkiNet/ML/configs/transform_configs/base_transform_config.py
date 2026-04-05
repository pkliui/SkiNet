from typing import Optional

from pydantic import BaseModel, Field


class BaseTransformConfig(BaseModel):
    """
    Base configuration for transformations.

    :attribute augmentation_required: Boolean flag indicating whether augmentation is required
            If False, only transforms under augmentations_off are applied.
    :attribute seed_value: Optional seed value for reproducibility of the
        transformations. In Albumentations, it is provided as a seed to the
        Compose object. Note that Albumentations uses its own internal random
        state that is completely independent from global random seeds.
    """
    augmentation_required: bool = Field(
        default=True, description="Whether to apply augmentations."
    )
    seed_value: Optional[int] = Field(
        default=None, description="Seed value for reproducibility."
    )
