
"""------------------------------------------------------------------TESTS for ensure_np_image---------------------------------------------------------------"""
import numpy as np
import torch
from PIL import Image
from SkiNet.Utils.image_utils import ensure_np_image

# Constants for input image dimensions
IMG_HEIGHT = 600
IMG_WIDTH = 600
IMG_CHANNELS = 3

def test_ensure_np_image_tensor_input():
    """
    Test that ensure_np_image converts a torch.Tensor to a numpy array"""
    tensor = torch.rand(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    out = ensure_np_image(tensor)
    assert isinstance(out, np.ndarray)

def test_ensure_np_image_pil_input():
    """
    Test that ensure_np_image converts a PIL image to a numpy array"""
    image = np.random.randint(0, 256, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8) 
    pil_img = Image.fromarray(image)
    out = ensure_np_image(pil_img)
    assert isinstance(out, np.ndarray)

def test_ensure_np_image_tensor_input_reshape():
    """
    Test that ensure_np_image converts a torch.Tensor to a numpy array
    and reshapes it to (H, W, C) if it is in (C, H, W) format"""
    tensor = torch.rand(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    out = ensure_np_image(tensor)
    assert isinstance(out, np.ndarray)
    assert out.shape == (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
