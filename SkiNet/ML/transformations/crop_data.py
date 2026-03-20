import logging

import torch

logger = logging.getLogger(__name__)

def crop_2d_image(image: torch.Tensor, crop_size: tuple[int, int]) -> list[slice]:
    """
    Returns slices for cropping spatial dimensions of a 2D image or 4D tensor.
    Handles images with shape (1, H, W, C) or (1, H, W), or (H, W, C), (H, W).

    :param crop_size: (crop_height, crop_width).
    :return: List of slice objects for spatial dimensions.
    """
    img_shape = image.shape

    # Identify spatial dimensions (always height and width)
    if len(img_shape) == 4:  # (batch, H, W, C)
        h, w = img_shape[1], img_shape[2]
        spatial_idxs = [1, 2]
    elif len(img_shape) == 3 and img_shape[-1] <= 3:  # (H, W, C)
        h, w = img_shape[0], img_shape[1]
        spatial_idxs = [0, 1]
    elif len(img_shape) == 3:  # (batch, H, W)
        h, w = img_shape[1], img_shape[2]
        spatial_idxs = [1, 2]
    elif len(img_shape) == 2:  # (H, W)
        h, w = img_shape[0], img_shape[1]
        spatial_idxs = [0, 1]
    else:
        logger.error(f"Unsupported image shape: {img_shape}")
        raise ValueError(f"Unsupported image shape: {img_shape}")

    crop_h, crop_w = crop_size
    if crop_h > h or crop_w > w:
        logger.error(f"Crop size {crop_size} exceeds image size {(h, w)}")
        raise ValueError(f"Crop size {crop_size} exceeds image size {(h, w)}")

    center_h = h // 2
    center_w = w // 2
    start_h = max(center_h - crop_h // 2, 0)
    end_h = start_h + crop_h
    start_w = max(center_w - crop_w // 2, 0)
    end_w = start_w + crop_w

    # Build slices for all dimensions, cropping only spatial dims
    slices = [slice(None)] * len(img_shape)
    slices[spatial_idxs[0]] = slice(start_h, end_h)
    slices[spatial_idxs[1]] = slice(start_w, end_w)
    return slices
