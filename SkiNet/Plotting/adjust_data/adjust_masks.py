import numpy as np

def adjust_mask_for_goimage(mask: np.array) -> np.array:
    """
    Adjust values of mask and make it look like RGB by duplicating its single channel

    param: mask
    """

    # Make RGB mask and adjust values to be able to display in go.Image
    mask = np.stack([mask.squeeze()]*3, axis=2)
    mask = (mask * 255).astype(int)
    return mask