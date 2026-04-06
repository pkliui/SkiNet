import albumentations as A
import pytest
import torch

from SkiNet.ML.configs.config_creator import PH2_UNet_ConfigCreator
from SkiNet.ML.datasets.sample_specs import Sample, SampleSpecs
from SkiNet.ML.transformations.transform_adapters import AlbumentationsSampleTransform
from SkiNet.ML.transformations.transform_data import _build_transform, get_transform_from_config


def _make_sample(height: int = 64, width: int = 64) -> Sample:
    return Sample(
        image=torch.ones((3, height, width), dtype=torch.uint8),
        mask=torch.ones((1, height, width), dtype=torch.uint8),
        specs=SampleSpecs(sample_id="sample-1", image_path="image.png", mask_path="mask.png"),
    )


@pytest.mark.parametrize(
    "transform_groups,expected_pipeline_types,expected_visualization_types",
    [
        (
            [
                [A.CenterCrop(height=32, width=32)],
                [A.HorizontalFlip(p=0.5)],
                [A.Normalize(), A.ToTensorV2(transpose_mask=True)],
            ],
            ["CenterCrop", "HorizontalFlip", "Normalize", "ToTensorV2"],
            ["CenterCrop", "HorizontalFlip"],
        ),
        (
            [
                [],
                [A.VerticalFlip(p=0.5)],
                [A.Normalize(), A.ToTensorV2(transpose_mask=True)],
            ],
            ["VerticalFlip", "Normalize", "ToTensorV2"],
            ["VerticalFlip"],
        ),
    ],
)
def test_build_transform_flattens_groups_and_excludes_postprocess_from_visualization(
    transform_groups: list[list[A.BasicTransform]],
    expected_pipeline_types: list[str],
    expected_visualization_types: list[str],
) -> None:
    """
    Test that _build_transform correctly flattens the list of transform groups into a single Albumentations pipeline,
    and that the visualization pipeline excludes postprocessing steps (like Normalize and ToTensorV2).
    """
    transform = _build_transform(transform_groups)

    assert isinstance(transform, AlbumentationsSampleTransform)
    assert [type(step).__name__ for step in transform.pipeline.transforms] == expected_pipeline_types
    assert transform.visualization_pipeline is not None  # for mypy
    assert [type(step).__name__ for step in transform.visualization_pipeline.transforms] == expected_visualization_types
    assert transform.expects_tensor_output is True


def test_build_transform_passes_compose_kwargs_to_main_and_visualization_pipelines() -> None:
    transform = _build_transform(
        [
            [A.CenterCrop(height=32, width=32)],
            [A.Normalize(), A.ToTensorV2(transpose_mask=True)],
        ],
        compose_kwargs={"seed": 7, "save_applied_params": True},
    )

    assert transform.pipeline.seed == 7
    assert transform.pipeline.save_applied_params is True
    assert transform.visualization_pipeline is not None
    assert transform.visualization_pipeline.seed == 7
    assert transform.visualization_pipeline.save_applied_params is True


@pytest.mark.parametrize(
    "transformconfig_kwargs,split_name,expected_pipeline_len,expected_visualization_len",
    [
        ({}, "train", 9, 7),
        ({}, "val", 3, 1),
        ({}, "test", 3, 1),
        (
            {
                "crop": {"crop_type": "center_crop", "size": (32, 24)},
                "spatial_augmentation": {
                    "horizontal_flip_apply": False,
                    "vertical_flip_apply": False,
                    "square_symmetry_apply": False,
                    "affine_apply": False,
                    "perspective_apply": False,
                },
                "photometric_augmentation": {"color_jitter_apply": False},
            },
            "train",
            3,  # crop + postprocess for train
            1,  # crop only for visualization
        ),
    ],
)
def test_get_transform_from_config_builds_expected_split_pipelines(
    transformconfig_kwargs: dict,
    split_name: str,
    expected_pipeline_len: int,
    expected_visualization_len: int,
) -> None:
    """
    Test that get_transform_from_config constructs the expected number of transforms in the pipeline for each split (train, val, test)
    based on the provided configuration.
    """
    cfg = PH2_UNet_ConfigCreator().create_config(transformconfig_kwargs=transformconfig_kwargs)

    transforms = get_transform_from_config(cfg)
    split_transform = getattr(transforms, split_name)

    assert isinstance(split_transform, AlbumentationsSampleTransform)
    assert len(split_transform.pipeline.transforms) == expected_pipeline_len
    assert split_transform.visualization_pipeline is not None
    assert len(split_transform.visualization_pipeline.transforms) == expected_visualization_len
    assert type(split_transform.pipeline.transforms[-1]).__name__ == "ToTensorV2"


def test_get_transform_from_config_forwards_seed_and_compose_kwargs() -> None:
    cfg = PH2_UNet_ConfigCreator().create_config(
        transformconfig_kwargs={
            "seed_value": 13,
            "compose_kwargs": {"save_applied_params": True, "strict": True},
        }
    )

    transforms = get_transform_from_config(cfg)

    assert transforms.train.pipeline.seed == 13
    assert transforms.train.pipeline.save_applied_params is True
    assert transforms.train.visualization_pipeline is not None
    assert transforms.train.visualization_pipeline.seed == 13


@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_get_transform_from_config_returns_callable_sample_transforms(split_name: str) -> None:
    """
    Test that the sample transforms returned by get_transform_from_config can be called on a sample and produce tensors of the expected shape,
    and that the sample specifications are preserved in the transformed output.
    """
    cfg = PH2_UNet_ConfigCreator().create_config(
        transformconfig_kwargs={
            "crop": {"crop_type": "center_crop", "size": (32, 24)},
            "spatial_augmentation": {
                "horizontal_flip_apply": False,
                "vertical_flip_apply": False,
                "square_symmetry_apply": False,
                "affine_apply": False,
                "perspective_apply": False,
            },
            "photometric_augmentation": {"color_jitter_apply": False},
        }
    )
    sample = _make_sample()

    transforms = get_transform_from_config(cfg)
    transformed = getattr(transforms, split_name)(sample)

    assert isinstance(transformed.image, torch.Tensor)
    assert isinstance(transformed.mask, torch.Tensor)
    assert transformed.image.shape == (3, 32, 24)
    assert transformed.mask.shape == (1, 32, 24)
    assert transformed.specs == sample.specs
