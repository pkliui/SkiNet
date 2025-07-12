import numpy as np
from typing import Union
import torch
import PIL

# possible input image data types that will be onverted to np.ndarray as per compatibility with albumentaions
ImageData = Union[np.ndarray,
                  torch.Tensor, 
                  PIL.Image.Image]


def ensure_np_image(x: ImageData) -> np.ndarray:
    """
    Ensure the input is a numpy array and that its shape is  (H, W, C) as required by Albumentations library.
    :param x: The input image that can be a numpy array or a tensor or a PIL image
    :return: A numpy array of (H, W, C)
    """
    if isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.array(x)
    # If array is 3D and channels are first and there are either 1 or 3 channels, 
    # move them to last
    if arr.ndim == 3 and arr.shape[0] in [1, 3] and arr.shape[0] < min(arr.shape[1:]):
        arr = np.transpose(arr, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    
    # If a bool mask, onvert to uint8 to e able to work with OpenCV and Albumentations
    if arr.dtype == bool:
        arr = arr.astype(np.uint8)
    return arr
