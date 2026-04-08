import albumentations as A
import numpy as np
import pytest
import torch

from SkiNet.ML.datasets.sample_specs import Sample, SampleSpecs
from SkiNet.ML.transformations.transform_adapters import AlbumentationsSampleTransform


def _make_sample(height: int = 64, width: int = 48) -> Sample:
    return Sample(
        image=torch.ones((3, height, width), dtype=torch.uint8),
        mask=torch.ones((1, height, width), dtype=torch.uint8),
        specs=SampleSpecs(sample_id="sample-1", image_path="image.png", mask_path="mask.png"),
    )


@pytest.mark.parametrize(
    "pipeline,expects_tensor_output,expected_image_type,expected_mask_type,expected_image_shape,expected_mask_shape",
    [
        (
            A.Compose([A.CenterCrop(height=32, width=24), A.ToTensorV2(transpose_mask=True)]),
            True,
            torch.Tensor,
            torch.Tensor,
            (3, 32, 24),  # CHW format for tensors after ToTensorV2 (default)
            (1, 32, 24),  # CHW format for tensors after ToTensorV2
        ),
        (
            A.Compose([A.CenterCrop(height=32, width=24)]),
            False,
            np.ndarray,
            np.ndarray,
            (32, 24, 3),  # HWC format for numpy arrays without ToTensorV2
            (32, 24, 1),  # HWC format for numpy arrays without ToTensorV2
        ),
    ],
)
def test_albumentations_sample_transform_returns_transformed_sample(
    pipeline: A.Compose,
    expects_tensor_output: bool,
    expected_image_type: type[torch.Tensor] | type[np.ndarray],
    expected_mask_type: type[torch.Tensor] | type[np.ndarray],
    expected_image_shape: tuple[int, ...],
    expected_mask_shape: tuple[int, ...],
) -> None:
    """
    AlbumentationsSampleTransform should return a new Sample with transformed image and mask values from the configured pipeline.

    Shape checks: ensure the tensor output is CHW and the numpy output is HWC, with the correct number of channels.
    """
    transform = AlbumentationsSampleTransform(
        pipeline=pipeline,
        visualization_pipeline=A.Compose([A.CenterCrop(height=32, width=24)]),
        expects_tensor_output=expects_tensor_output,
    )
    sample = _make_sample()

    transformed = transform(sample)

    assert isinstance(transformed.image, expected_image_type)
    assert isinstance(transformed.mask, expected_mask_type)
    assert transformed.image.shape == expected_image_shape
    assert transformed.mask.shape == expected_mask_shape
    assert transformed.specs == sample.specs


@pytest.mark.parametrize(
    "pipeline",
    [
        A.Compose([A.CenterCrop(height=32, width=24)]),
    ],
)
def test_albumentations_sample_transform_raises_when_tensor_output_is_expected(
    pipeline: A.Compose,
) -> None:
    """
    AlbumentationsSampleTransform should raise TypeError when tensor output is required but the pipeline returns numpy arrays.
    """
    transform = AlbumentationsSampleTransform(pipeline=pipeline, expects_tensor_output=True)
    sample = _make_sample()

    with pytest.raises(TypeError, match="Transform pipeline must output torch tensors"):
        transform(sample)


@pytest.mark.parametrize(
    "visualization_pipeline",
    [
        A.Compose([A.CenterCrop(height=32, width=24)]),
        None,
    ],
)
def test_without_postprocess_returns_visualization_transform(
    visualization_pipeline: A.Compose | None,
) -> None:
    """
    without_postprocess() should return a transform that uses the visualization pipeline and disables tensor-output enforcement.
    """
    pipeline = A.Compose([A.CenterCrop(height=32, width=24), A.ToTensorV2(transpose_mask=True)])
    transform = AlbumentationsSampleTransform(
        pipeline=pipeline,
        visualization_pipeline=visualization_pipeline,
        expects_tensor_output=True,
    )

    without_postprocess = transform.without_postprocess()

    expected_pipeline = visualization_pipeline if visualization_pipeline is not None else pipeline

    assert without_postprocess.pipeline is expected_pipeline
    assert without_postprocess.visualization_pipeline is expected_pipeline
    assert without_postprocess.expects_tensor_output is False


def test_without_postprocess_returns_numpy_outputs_when_visualization_pipeline_omits_postprocess() -> None:
    """
    without_postprocess() should allow visualization-only pipelines that return numpy arrays instead of tensors.
    """
    transform = AlbumentationsSampleTransform(
        pipeline=A.Compose([A.CenterCrop(height=32, width=24), A.ToTensorV2(transpose_mask=True)]),
        visualization_pipeline=A.Compose([A.CenterCrop(height=32, width=24)]),
        expects_tensor_output=True,
    )
    sample = _make_sample()

    transformed = transform.without_postprocess()(sample)

    assert isinstance(transformed.image, np.ndarray)
    assert isinstance(transformed.mask, np.ndarray)
    assert transformed.image.shape == (32, 24, 3)
    assert transformed.mask.shape == (32, 24, 1)
