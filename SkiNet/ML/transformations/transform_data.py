from dataclasses import dataclass
from typing import Any

import albumentations as A

from SkiNet.ML.configs.experiment_config import ExperimentConfig
from SkiNet.ML.transformations.transform_adapters import (
    AlbumentationsSampleTransform,
    SampleTransformAdapter,
)
from SkiNet.ML.transformations.transform_pipelines import (
    get_crop_transforms,
    get_photometric_transforms,
    get_postprocess_transforms,
    get_spatial_transforms,
)


@dataclass(frozen=True)
class TransformsContainer:
    """
    Transformation container for train, val, and test modes.
    Each mode has its own SampleTransformAdapter, which can be an
    AlbumentationsSampleTransform or any other implementation of the
    SampleTransformAdapter protocol.
    """

    train: SampleTransformAdapter
    val: SampleTransformAdapter
    test: SampleTransformAdapter


def _flatten(transform_groups: list[list[A.BasicTransform]]) -> list[A.BasicTransform]:
    return [transform for group in transform_groups for transform in group]


def _resolve_compose_kwargs(cfg: ExperimentConfig) -> dict[str, Any]:
    """
    Resolve the keyword arguments for the Albumentations Compose function based on the experiment configuration.

    :param cfg: The experiment configuration containing the transformation settings, including any additional keyword arguments
    for the Compose function.
    :return: A dictionary containing the keyword arguments for the transformation pipelines.
    """
    compose_kwargs = dict(cfg.transformconfig.compose_kwargs)
    # If a seed value is provided in the configuration and not already included in the compose_kwargs,
    # add it to compose kwargs to ensure reproducibility of the transformations.
    if cfg.transformconfig.seed_value is not None and "seed" not in compose_kwargs:
        compose_kwargs["seed"] = cfg.transformconfig.seed_value

    return compose_kwargs


def _build_transform(transform_groups: list[list[A.BasicTransform]],
                     compose_kwargs: dict[str, Any] | None = None) -> AlbumentationsSampleTransform:
    """
    Build an AlbumentationsSampleTransform from the provided transform groups.
    The visualization pipeline includes all transforms except the last group,
    which is assumed to contain post-processing transforms
    that should not be applied during visualization.

    :param transform_groups: A list of lists of Albumentations transforms,
        where each inner list represents a group of transforms that should be
        applied together (e.g., crop, spatial, photometric, postprocess).
    """
    visualisation_transforms = _flatten(transform_groups[:-1])
    all_transforms = _flatten(transform_groups)
    compose_kwargs = compose_kwargs or {}

    return AlbumentationsSampleTransform(pipeline=A.Compose(all_transforms, **compose_kwargs),
                                         visualization_pipeline=A.Compose(visualisation_transforms, **compose_kwargs),
                                         expects_tensor_output=True)


def get_transform_from_config(cfg: ExperimentConfig) -> TransformsContainer:
    """
    Construct split-specific sample transforms from the experiment configuration.

    :param cfg: The experiment configuration containing the transformation
        settings for cropping, spatial augmentation, photometric augmentation,
        and post-processing. If certain transformations are not specified in
        the configuration, default values specified in the relevant
        configuration classes (e.g., CropConfig, SpatialAugmentConfig,
        PhotoAugmentConfig) will be used.

    :return: A TransformsContainer object containing the train, val, and test sample transforms constructed based on the provided configuration.
    """

    crop_transforms = get_crop_transforms(cfg.transformconfig.crop)
    spatial_transforms = get_spatial_transforms(
        cfg.transformconfig.spatial_augmentation)
    photometric_transforms = get_photometric_transforms(
        cfg.transformconfig.photometric_augmentation)
    postprocess_transforms = get_postprocess_transforms(cfg.transformconfig)
    compose_kwargs = _resolve_compose_kwargs(cfg)

    train_pipeline = _build_transform(
        [
            crop_transforms,
            spatial_transforms,
            photometric_transforms,
            postprocess_transforms,
        ],
        compose_kwargs=compose_kwargs,
    )
    val_pipeline = _build_transform(
        [
            crop_transforms,
            [],
            [],
            postprocess_transforms,
        ],
        compose_kwargs=compose_kwargs,
    )
    test_pipeline = _build_transform(
        [
            crop_transforms,
            [],
            [],
            postprocess_transforms,
        ],
        compose_kwargs=compose_kwargs,
    )

    return TransformsContainer(train=train_pipeline,
                               val=val_pipeline,
                               test=test_pipeline)
