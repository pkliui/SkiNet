#seed value for albumentations
SEED_VALUE = 42
#image dimenstions for albumentations.CenterCrop
CROP_HEIGHT = 400
CROP_WIDTH = 400
# as per  config_test_transform_data
CROP_HEIGHT_PY_CONFIG = 500
CROP_WIDTH_PY_CONFIG = 500
# as per TRANSFORMATION_CONFIGS_YAML_PATH 
CROP_HEIGHT_YAML_CONFIG = 400
CROP_WIDTH_YAML_CONFIG = 400
# Constants for input image dimensions
IMG_HEIGHT = 600
IMG_WIDTH = 600
IMG_CHANNELS = 3


"""------------------------------------------------------------------TESTS for apply_transforms---------------------------------------------------------------"""
import numpy as np
import pytest
import torch
import albumentations as A
from PIL import Image

from SkiNet.ML.transformations.transform_data import TransformData

transforms = A.Compose([
    A.CenterCrop(height=CROP_HEIGHT,
                 width=CROP_WIDTH),
    A.ToTensorV2()
])

@pytest.fixture
def transform_data():
    """
    Fixture for TransformData with a transform that does a random resized crop.
    """
    return TransformData(transforms)

def test_apply_transforms_numpy_array(transform_data):
    """
    Test that apply_transforms a numpy array input and returns a torch.Tensor
    """
    image = np.random.randint(0, 256, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

    out= transform_data.apply_transforms(image=image)
    assert isinstance(out['image'], torch.Tensor) 


def test_apply_transforms_pil(transform_data):
    """
    Test that apply_transforms a PIL image input and returns a torch.Tensor
    """
    image = np.random.randint(0, 256, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    pil_img = Image.fromarray(image)

    out = transform_data.apply_transforms(pil_img)
    assert isinstance(out['image'], torch.Tensor)

def test_apply_transforms_image_mask_tensor_input(transform_data):
    """
    Test that apply_transforms transforms an image and a mask both of which are tensors  and returns transformed torch.Tensors
    of the same dtype and expected value as per transform
    """
    image = np.ones((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    mask = np.ones((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    transformed = transform_data.apply_transforms(image=image, mask=mask)
    image_transformed = transformed['image']
    mask_transformed = transformed['mask']
    assert isinstance(image_transformed, torch.Tensor)
    assert isinstance(mask_transformed, torch.Tensor)
    assert torch.allclose(image_transformed, torch.ones(IMG_CHANNELS, CROP_HEIGHT, CROP_WIDTH,  dtype=torch.uint8))
    assert torch.allclose(mask_transformed, torch.ones(CROP_HEIGHT, CROP_WIDTH,  dtype=torch.uint8))

def test_apply_transforms_image_mask_PIL(transform_data):
    """
    Test that apply_transforms transforms an image and a mask both of which are PIL images  and returns transformed torch.Tensors
    of the same dtype and expected value as per transform
    """
    image = np.ones((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    mask = np.ones((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

    # convert to PIL
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)

    # transform
    transformed = transform_data.apply_transforms(image=image, mask=mask)
    image_transformed = transformed['image']
    mask_transformed = transformed['mask']
    assert isinstance(image_transformed, torch.Tensor)
    assert isinstance(mask_transformed, torch.Tensor)
    assert torch.allclose(image_transformed, torch.ones(IMG_CHANNELS, CROP_HEIGHT, CROP_WIDTH,  dtype=torch.uint8))
    assert torch.allclose(mask_transformed, torch.ones(CROP_HEIGHT, CROP_WIDTH,  dtype=torch.uint8))


"""------------------------------------------------------------------TESTS for TransformData---------------------------------------------------------------"""
from SkiNet.ML.transformations.transform_data import TransformData

def test_can_pass_pipeline_to_transformdata():
    """
    Test that we can pass a albumentations pipeline to TransformData and it is stored correctly.
    """
    pipeline = A.Compose([A.ToTensorV2()])
    transform = TransformData(pipeline)
    assert hasattr(transform, "pipeline")
    assert transform.pipeline is pipeline

def test_transformdata_pipeline_is_callable():
    """
    Test that the TransformData instance with a pipeline can be called and returns a transformed tensor."""
    pipeline = A.Compose([A.ToTensorV2()])
    transform = TransformData(pipeline)
    
    image = np.ones((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    # Should not raise
    out = transform(image=image)
    
    assert isinstance(out['image'], torch.Tensor)

def test_transformdata_pipeline_with_pil():
    """
    Test that the TransformData instance with a pipeline can be called with a PIL image and returns a transformed tensor.   
    """
    pipeline = A.Compose([A.ToTensorV2()])
    transform = TransformData(pipeline)

    image = np.ones((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    pil_img = Image.fromarray(image)

    out = transform(pil_img)
    assert isinstance(out['image'], torch.Tensor)


"""------------------------------------------------------------------TESTS for make_transform_from_config using default YACS config ---------------------------------------------------------------"""

from SkiNet.ML.transformations.transform_data import TransformData, make_transform_from_config
from Tests.ML.configs.transformation_configs_paths_for_test import config_test_transform_data

@pytest.fixture
def explicit_transforms():
    explicit_transforms_from_config = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(rotate=(-90, 90), translate_percent=(0.1, 0.1),  shear=(0, 30)),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,  p=0.5),
    A.CenterCrop(height=CROP_HEIGHT_PY_CONFIG, width=CROP_WIDTH_PY_CONFIG),
    A.ToTensorV2()], 
    seed = SEED_VALUE)
    return explicit_transforms_from_config

@pytest.fixture
def explicit_transforms_augmentation_off():
    explicit_transforms_from_config = [
    A.CenterCrop(height=CROP_HEIGHT_PY_CONFIG, width=CROP_WIDTH_PY_CONFIG),
    A.ToTensorV2()
]
    return explicit_transforms_from_config

def test_make_transform_from_config_PIL_image(explicit_transforms) -> None:
    """
    Test that make_transform_from_config returns the correctly transformed input
    that is a PIL image
    """
    #
    # return the transformation pipeline using the provided configuration
    transform_from_config = make_transform_from_config(
        config_test_transform_data,
        augmentation_required=True,
        seed_value=SEED_VALUE
    )
    # input as a PIL image
    image_input = np.ones((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    image_input = Image.fromarray(image_input)

    # get the transformed image
    transformed = transform_from_config(image=image_input)
    transformed_image = transformed['image']
    assert isinstance(transformed_image, torch.Tensor) # torch.Tensor

    # get the transformed image using the expected pipeline above
    expected_transformed_image = explicit_transforms(image=np.array(image_input))['image']
    import logging
    logging.getLogger(__name__).debug(f"Expected transformed image: {expected_transformed_image.shape}")

    assert expected_transformed_image.shape == transformed_image.shape
    assert expected_transformed_image.shape == (IMG_CHANNELS, CROP_HEIGHT_PY_CONFIG, CROP_WIDTH_PY_CONFIG)
    assert expected_transformed_image.dtype == transformed_image.dtype
    assert isinstance(expected_transformed_image, torch.Tensor) # torch.Tensor
    # assert both results are the same
    assert torch.isclose(expected_transformed_image, transformed_image).all()

"""------------------------------------------------------------------TESTS for make_transform_from_config using YAML config ---------------------------------------------------------------"""


# obtain the transform
from SkiNet.ML.transformations.transform_data import make_transform_from_config
# import YAML config
from Tests.ML.configs.transformation_configs_paths_for_test import config_yaml_test_transform_data


@pytest.fixture
def explicit_transforms_YAML():
    return A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(rotate=(-90, 90), translate_percent=(0.1, 0.1),  shear=(-20, 20)),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,  p=0.5),
    A.CenterCrop(height=CROP_HEIGHT_YAML_CONFIG, width=CROP_WIDTH_YAML_CONFIG),
    A.ToTensorV2()], 
    seed = SEED_VALUE)

def test_make_transform_from_configYAML_PIL_image(explicit_transforms_YAML):
    """
    Test that a YAML configuration that overrides the default transdformations_config.py 
    returns the correctly transformed input that is a PIL image
    """

    transform_from_YAMLconfig = make_transform_from_config(
        config_yaml_test_transform_data,
        augmentation_required=True,
        seed_value=SEED_VALUE)

    # input as a PIL image
    image_input = np.ones((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    image_input = Image.fromarray(image_input)

    # get the transformed image
    transformed = transform_from_YAMLconfig(image=image_input)
    transformed_image = transformed['image']
    assert isinstance(transformed_image, torch.Tensor) # torch.Tensor

    # get the transformed image using the expected pipeline above
    expected_transformed_image = explicit_transforms_YAML(image=np.array(image_input))['image']

    assert expected_transformed_image.shape == transformed_image.shape
    assert expected_transformed_image.dtype == transformed_image.dtype
    assert isinstance(expected_transformed_image, torch.Tensor) # torch.Tensor
    # assert both results are the same
    assert torch.isclose(expected_transformed_image, transformed_image).all()