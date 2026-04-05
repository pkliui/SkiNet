from typing import Union

import numpy as np
import torch


def convert_to_hwc_numpy(data_item: Union[torch.Tensor, np.ndarray]) -> Union[np.ndarray, None]:
    """
    Convert image or mask data to a numpy array compatible with Albumentations.

    Accepts:
    - torch.Tensor in CHW, HWC, or HW layout
    - np.ndarray in CHW, HWC, or HW layout

    Returns:
    - HWC numpy array for channelled inputs
    - HW numpy array for single-plane inputs
    """

    def _convert_ndarray(x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            return np.expand_dims(x, axis=-1)  # HW -> HW1
        if x.ndim != 3:
            raise ValueError(
                f"Expected data item to have shape (H, W), (C, H, W), or (H, W, C), but got {x.shape}"
            )
        # CHW -> HWC
        if x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
            return np.transpose(x, (1, 2, 0))

        # HWC with 1 or 3 channels is already in the correct format
        if x.shape[-1] in (1, 3):
            return x

        raise ValueError(
            f"Expected 3D data item to use 1 or 3 channels in first or last dimension, but got {x.shape}"
        )

    if isinstance(data_item, torch.Tensor):
        return _convert_ndarray(data_item.detach().cpu().numpy())

    if isinstance(data_item, np.ndarray):
        return _convert_ndarray(data_item)
