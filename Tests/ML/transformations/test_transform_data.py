import logging
import random

import numpy as np
import PIL
import pytest
import torch
import torchvision.transforms.v2 as T
from PIL import Image

from SkiNet.ML.transformations.transform_data import (
    TransformData, make_transform_from_config)

IMAGE_SIZE = (600, 600)
RESIZE_SIZE = (512, 512)

"""------------------------------------------------------------------TESTS for ensure_tv_image---------------------------------------------------------------"""



import numpy as np
import pytest
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torchvision.tv_tensors import Image as TVImage

from SkiNet.ML.transformations.transform_data import TransformData


class DummyTransform:
    def __call__(self, x):
        return x

@pytest.fixture
def transform_data():
    """
    Fixture for TransformData.
    """
    return TransformData(DummyTransform())

def test_ensure_tv_image_tensor(transform_data):
    """
    Test that ensure_tv_image converts a torch.Tensor to a TVImage"""
    tensor = torch.rand(3, 32, 32)
    out = transform_data.ensure_tv_image(tensor)
    assert isinstance(out, TVImage)
    assert out.shape == tensor.shape

def test_ensure_tv_image_pil(transform_data):
    """
    Test that ensure_tv_image converts a PIL.Image.Image to a TVImage
    """
    arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    pil_img = Image.fromarray(arr)
    out = transform_data.ensure_tv_image(pil_img)
    assert isinstance(out, TVImage)
    assert out.shape == (3, 32, 32)

def test_ensure_tv_image_tvimage(transform_data):
    """
    Test that ensure_tv_image returns the same TVImage object if the input is already a TVImage.
    """
    tensor = torch.rand(3, 32, 32)
    tv_img = T.ToImage()(tensor)
    out = transform_data.ensure_tv_image(tv_img)
    assert isinstance(out, TVImage)
    assert out is tv_img  # Should return the same object, not a new one


"""------------------------------------------------------------------TESTS for apply_transforms---------------------------------------------------------------"""


import numpy as np
import pytest
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torchvision.transforms import v2
from torchvision.tv_tensors import Image as TVImage

from SkiNet.ML.transformations.transform_data import TransformData

transforms = v2.Compose([
    v2.Resize(size=RESIZE_SIZE, antialias=True),
    v2.ToDtype(torch.float32, scale=True)
])

@pytest.fixture
def transform_data():
    """
    Fixture for TransformData with a transform that does a random resized crop.
    """
    return TransformData(transforms)

def test_apply_transforms_tensor(transform_data):
    """
    Test that apply_transforms a tensor input and returns a torch.Tensor
    """
    tensor = torch.zeros(3, 32, 32)
    out = transform_data.apply_transforms(tensor)
    assert isinstance(out, torch.Tensor) #By default, operations on TVTensor objects will return a pure Tensor
    assert torch.allclose(out,  torch.zeros(3, *RESIZE_SIZE))  

def test_apply_transforms_pil(transform_data):
    """
    Test that apply_transforms a PIL image input and returns a torch.Tensor
    """
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    pil_img = Image.fromarray(arr)
    out = transform_data.apply_transforms(pil_img)
    assert isinstance(out, torch.Tensor)
    assert torch.allclose(out,  torch.zeros(3, *RESIZE_SIZE))  


def test_apply_transforms_tvimage(transform_data):
    """
    Test that apply_transforms a TVImage image input and returns a torch.Tensor
    """
    tensor = torch.zeros(3, 32, 32)
    tv_img = T.ToImage()(tensor)
    out = transform_data.apply_transforms(tv_img)
    assert isinstance(out, torch.Tensor)
    assert torch.allclose(out, torch.zeros(3, *RESIZE_SIZE))


def test_apply_transforms_tuple_tensor(transform_data):
    """
    Test that apply_transforms a tuple of tensor images input and returns a tuple of transformed torch.Tensors, 
    That is because we ensure both of them as converted to TVImage before applying the transforms
    """
    tensor1 = torch.zeros(3, 32, 32)
    tensor2 = torch.zeros(3, 32, 32)
    out1, out2 = transform_data.apply_transforms((tensor1, tensor2))
    assert isinstance(out1, torch.Tensor)
    assert isinstance(out2, torch.Tensor)
    assert torch.allclose(out1, torch.zeros(3, *RESIZE_SIZE))
    assert torch.allclose(out2, torch.zeros(3, *RESIZE_SIZE))

def test_apply_transforms_tuple_tvimage(transform_data):
    """
    Test that apply_transforms a tuple of TVImage images input and returns a tuple of transformed torch.Tensors
    """
    tensor1 = torch.zeros(3, 32, 32)
    tensor2 = torch.zeros(3, 32, 32)
    tv_img1 = T.ToImage()(tensor1)
    tv_img2 = T.ToImage()(tensor2)
    out1, out2 = transform_data.apply_transforms((tv_img1, tv_img2))
    assert isinstance(out1, torch.Tensor)
    assert isinstance(out2, torch.Tensor)
    assert torch.allclose(out1, torch.zeros(3, *RESIZE_SIZE))
    assert torch.allclose(out2, torch.zeros(3, *RESIZE_SIZE))



"""------------------------------------------------------------------TESTS for TransformData---------------------------------------------------------------"""


from SkiNet.ML.transformations.transform_data import TransformData


def test_can_pass_pipeline_to_transformdata():
    """
    Test that we can pass a^ torchvision.transforms.v2 pipeline to TransformData and it is stored correctly.
    """
    pipeline = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    transform = TransformData(pipeline)
    assert hasattr(transform, "pipeline")
    assert transform.pipeline is pipeline

def test_transformdata_pipeline_is_callable():
    """
    Test that the TransformData instance with a pipeline can be called and returns a transformed tensor."""
    pipeline = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    transform = TransformData(pipeline)
    
    tensor = torch.zeros(3, 32, 32)
    # Should not raise
    out = transform(tensor)
    
    assert isinstance(out, torch.Tensor)

def test_transformdata_pipeline_with_pil():
    """
    Test that the TransformData instance with a pipeline can be called with a PIL image and returns a transformed tensor.   
    """
    pipeline = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    transform = TransformData(pipeline)

    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    pil_img = Image.fromarray(arr)

    out = transform(pil_img)
    assert isinstance(out, torch.Tensor)


"""------------------------------------------------------------------TESTS for make_transform_from_config---------------------------------------------------------------"""

from SkiNet.ML.transformations.transform_data import (
    TransformData, make_transform_from_config)
from Tests.ML.configs.test_transformation_configs import \
    config_test_transform_data


@pytest.fixture
def explicit_transforms():
    explicit_transforms_from_config = [
    T.RandomAffine(degrees=30, translate=(0.05, 0.05), scale = (0.1, 0.3), shear=30),
    T.RandomRotation(degrees = 30),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.ElasticTransform(sigma=5.0, alpha=50),
    T.RandomPerspective(distortion_scale = 0.5, p = 0.5),
    T.RandomEqualize(p=0.5),
    T.ColorJitter(saturation=0.5, brightness=0.5, contrast=0.5, hue = 0.5),
    T.CenterCrop(size=(500, 500)),
    T.ToDtype(torch.float32, scale=True)
]
    return explicit_transforms_from_config

@pytest.fixture
def explicit_transforms_augmentation_off():
    explicit_transforms_from_config = [
    T.CenterCrop(size=(500, 500)),
    T.ToDtype(torch.float32, scale=True)
]
    return explicit_transforms_from_config

def test_make_transform_from_config_PIL_image(explicit_transforms) -> None:
    """
    Test that make_transform_from_config returns the correctly transformed input
    that is a PIL image
    """
    #
    # return the transformation pipeline by make_transform_from_config using the provided configuration
    transform_from_config = make_transform_from_config(
        config_test_transform_data,
        augmentation_required=True
    )
    
    # list all transforms from config_test_transform_data
    explicit_transforms_from_config = explicit_transforms
    # input as a PIL image
    image_input = torch.randint(0, 256, size=(3, *IMAGE_SIZE), dtype=torch.uint8)
    image_input = T.ToPILImage()(image_input)

    # get the transformed image using transforms returned by make_transform_from_config
    np.random.seed(3)
    torch.manual_seed(3)
    random.seed(3)
    # transformed_image will be of TVImage type
    transformed_image = transform_from_config(image_input)
    logging.getLogger(__name__).debug(type(transformed_image))
    assert isinstance(transformed_image, TVImage)
    assert isinstance(transformed_image, torch.Tensor) # TVImage is a subclass of torch.Tensor

    # get the transformed image using the pipeline above
    np.random.seed(3)
    torch.manual_seed(3)
    random.seed(3)
    # convert to ToImage first, because this is done so in TransformData for all inputs that are not TVImage
    expected_transformed_image = T.ToImage()(image_input) 
    for transform in explicit_transforms_from_config:
        expected_transformed_image = transform(expected_transformed_image)

    # assert both results are the same
    assert torch.isclose(expected_transformed_image, transformed_image).all()




def test_make_transform_from_config_torch_tensor(explicit_transforms) -> None:
    """
    Test that make_transform_from_config returns the correctly transformed input
    that is a torch tensor
    """
    #
    # return the transformation pipeline by make_transform_from_config using the provided configuration
    transform_from_config = make_transform_from_config(
        config_test_transform_data,
        augmentation_required=True
    )
    
    # list all transforms from config_test_transform_data
    explicit_transforms_from_config = explicit_transforms
    # input as a torch tensor
    image_input = torch.randint(0, 256, size=(3, *IMAGE_SIZE), dtype=torch.uint8)

    # get the transformed image using transforms returned by make_transform_from_config
    np.random.seed(3)
    torch.manual_seed(3)
    random.seed(3)
    # transformed_image will be of TVImage type
    transformed_image = transform_from_config(image_input)
    logging.getLogger(__name__).debug(type(transformed_image))
    assert isinstance(transformed_image, TVImage)
    assert isinstance(transformed_image, torch.Tensor) # TVImage is a subclass of torch.Tensor

    # get the transformed image using the pipeline above
    np.random.seed(3)
    torch.manual_seed(3)
    random.seed(3)
    # convert to ToImage first, because this is done so in TransformData for all inputs that are not TVImage
    expected_transformed_image = T.ToImage()(image_input) 
    for transform in explicit_transforms_from_config:
        expected_transformed_image = transform(expected_transformed_image)

    # assert both results are the same
    assert torch.isclose(expected_transformed_image, transformed_image).all()


def test_make_transform_from_config_tvimage(explicit_transforms) -> None:
    """
    Test that make_transform_from_config returns the correctly transformed input
    that is a TV Image
    """
    #
    # return the transformation pipeline by make_transform_from_config using the provided configuration
    transform_from_config = make_transform_from_config(
        config_test_transform_data,
        augmentation_required=True
    )
    
    # list all transforms from config_test_transform_data
    explicit_transforms_from_config = explicit_transforms
    # input as a torch tensor
    image_input = torch.randint(0, 256, size=(3, *IMAGE_SIZE), dtype=torch.uint8)
    image_input = T.ToImage()(image_input)

    # get the transformed image using transforms returned by make_transform_from_config
    np.random.seed(3)
    torch.manual_seed(3)
    random.seed(3)
    # transformed_image will be of TVImage type
    transformed_image = transform_from_config(image_input)
    logging.getLogger(__name__).debug(type(transformed_image))
    assert isinstance(transformed_image, TVImage)
    assert isinstance(transformed_image, torch.Tensor) # TVImage is a subclass of torch.Tensor

    # get the transformed image using the pipeline above
    np.random.seed(3)
    torch.manual_seed(3)
    random.seed(3)
    # inputs is already a TVImage
    expected_transformed_image = image_input
    for transform in explicit_transforms_from_config:
        expected_transformed_image = transform(expected_transformed_image)

    # assert both results are the same
    assert torch.isclose(expected_transformed_image, transformed_image).all()




def test_make_transform_from_config_PIL_image_tuple(explicit_transforms) -> None:
    """
    Test that make_transform_from_config returns the correctly transformed input
    that is a tuple of two PIL images - image and mask
    """
    #
    # return the transformation pipeline by make_transform_from_config using the provided configuration
    transform_from_config = make_transform_from_config(
        config_test_transform_data,
        augmentation_required=True
    )
    
    # list all transforms from config_test_transform_data
    explicit_transforms_from_config = explicit_transforms
    # input as a PIL image
    image_input = torch.randint(0, 256, size=(3, *IMAGE_SIZE), dtype=torch.uint8)
    mask_input = torch.randint(0, 256, size=(3, *IMAGE_SIZE), dtype=torch.uint8)
    mask_input[100:200, 100:200] = 0
    image_input = T.ToPILImage()(image_input)
    mask_input = T.ToPILImage()(mask_input)

    # get the transformed image using transforms returned by make_transform_from_config
    np.random.seed(3)
    torch.manual_seed(3)
    random.seed(3)
    # transformed_image will be of TVImage type
    (transformed_image, transformed_mask) = transform_from_config((image_input, mask_input))
    logging.getLogger(__name__).debug(type(transformed_image))
    assert isinstance(transformed_image, TVImage)
    assert isinstance(transformed_image, torch.Tensor) # TVImage is a subclass of torch.Tensor
    assert isinstance(transformed_mask, TVImage)
    assert isinstance(transformed_mask, torch.Tensor) # TVImage is a subclass of torch.Tensor

    # get the transformed image using the pipeline above
    np.random.seed(3)
    torch.manual_seed(3)
    random.seed(3)
    # convert to ToImage first, because this is done so in TransformData for all inputs that are not TVImage
    expected_transformed_image = T.ToImage()(image_input) 
    expected_transformed_mask = T.ToImage()(mask_input) 
    for transform in explicit_transforms_from_config:
        expected_transformed_image, expected_transformed_mask = transform((expected_transformed_image, expected_transformed_mask))

    # assert both results are the same
    assert torch.isclose(expected_transformed_image, transformed_image).all()
    assert torch.isclose(expected_transformed_mask, transformed_mask).all()



def test_make_transform_from_config_PIL_image_augmentations_off(explicit_transforms_augmentation_off) -> None:
    """
    Test that make_transform_from_config returns the correctly transformed input
    that is a PIL image - augmentations off
    """
    #
    # return the transformation pipeline by make_transform_from_config using the provided configuration
    transform_from_config = make_transform_from_config(
        config_test_transform_data,
        augmentation_required=False
    )
    
    # list all transforms from config_test_transform_data
    explicit_transforms_from_config = explicit_transforms_augmentation_off
    # input as a PIL image
    image_input = torch.randint(0, 256, size=(3, *IMAGE_SIZE), dtype=torch.uint8)
    image_input = T.ToPILImage()(image_input)

    # get the transformed image using transforms returned by make_transform_from_config
    np.random.seed(3)
    torch.manual_seed(3)
    random.seed(3)
    # transformed_image will be of TVImage type
    transformed_image = transform_from_config(image_input)
    logging.getLogger(__name__).debug(type(transformed_image))
    assert isinstance(transformed_image, TVImage)
    assert isinstance(transformed_image, torch.Tensor) # TVImage is a subclass of torch.Tensor

    # get the transformed image using the pipeline above
    np.random.seed(3)
    torch.manual_seed(3)
    random.seed(3)
    # convert to ToImage first, because this is done so in TransformData for all inputs that are not TVImage
    expected_transformed_image = T.ToImage()(image_input) 
    for transform in explicit_transforms_from_config:
        expected_transformed_image = transform(expected_transformed_image)

    # assert both results are the same
    assert torch.isclose(expected_transformed_image, transformed_image).all()
