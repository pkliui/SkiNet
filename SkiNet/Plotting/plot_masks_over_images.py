"""
Plot an arbitrary number of images and masks, provided full path to them. Overlap each mask over the corresponding image.
Adapted from https://pytorch.org/vision/stable/auto_examples/others/plot_visualization_utils.html#sphx-glr-auto-examples-others-plot-visualization-utils-py
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from IPython.display import display
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks

from SkiNet.Utils.dev_utils import is_running_in_docker

# Set matplotlib backend based on environment
if is_running_in_docker():
    # to plot in docker
    import matplotlib # isort: skip
    matplotlib.use("Agg")


def plot_masks_over_images(images: Union[torch.Tensor, List[Image.Image]], 
                           masks: Union[torch.Tensor, List[Image.Image]], 
                           alpha: Optional[float] = 0.3,
                           colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = "white",
                           max_cols: Optional[int] = 2):
    """
    Plot masks over images with transparency.
    
    :param images:
        A batched tensor or a list of PIL images of shape (3, H, W) where 3 is the number of channels, 
        and H, W are the height and width of the images, and of dtype uint8.
    :param masks:
        A batched tensor or a list of PIL image of shape (H, W) or (num_masks, H, W) where num_masks is the number of masks,
        and H, W are the height and width of the masks, and of dtype bool.
    :param alpha:
        Transparency level of the masks. A value between 0 (fully transparent) and 1 (fully opaque). 
        Default is 0.3.
    :param colors:
        Colors for the masks. Can be a single color (e.g., "red" or (255, 0, 0)) or a list of colors for multiple masks. 
        Default is "white".
    :param max_cols:
        Maximum number of columns in the grid. Images will wrap to the next row if this limit is exceeded. 
        Default is 2.

    Notes:
    ------
    - This function uses `torchvision.utils.draw_segmentation_masks` for rendering masks. 
      See the official documentation for more details: 
      https://pytorch.org/vision/stable/utils.html#torchvision.utils.draw_segmentation_masks
    - Images and masks are transformed to uint8 tensors internally if needed.

    Example of using batched tensors:
    --------
    >>> images = torch.randint(0, 256, (4, 3, 128, 128), dtype=torch.uint8)
    >>> masks = torch.randint(0, 2, (4, 128, 128), dtype=torch.bool)
    >>> plot_masks_over_images(images, masks, alpha=0.5, colors="white", max_cols=2)
    """
    

    def _to_tensor_uint8(input_image: Image.Image) -> torch.Tensor:
        # Convert a PIL image to a uint8 tensor as required by draw_segmentation_masks
        if isinstance(input_image, Image.Image):
            transform = transforms.Compose([
                transforms.ToTensor(),  # Converts to torch.Tensor of float32 (0-1)
                transforms.Lambda(lambda x: (x * 255).byte())  # Scale & convert to uint8
            ])
        elif isinstance(input_image, torch.Tensor):
            transform = transforms.Compose([
                transforms.Lambda(lambda x: (x * 255).byte())  # Scale & convert to uint8
            ])
        else:
            raise TypeError(f"Unsupported input type: {type(input_image)}. Expected PIL.Image.Image or torch.Tensor.")
        return transform(input_image)
    

    images = [_to_tensor_uint8(img) for img in images]
    masks = [_to_tensor_uint8(mask) for mask in masks]
    # Convert masks to bool tensors
    masks = [mask.to(torch.bool) if mask.dtype != torch.bool else mask for mask in masks]

    # gather the torch.Tensor output images overlayed with masks and pass them to show for display
    images_with_masks = [draw_segmentation_masks(image, masks=mask, alpha=alpha, colors=colors) for image, mask in zip(images, masks)]
    show(images_with_masks, max_cols=max_cols)


def show(input_images: List[torch.Tensor],
         max_cols: int = 2):
    """
    Show images provided as a list of torch tensors, with automatic wrapping of images
    into multiple rows if needed.

    Adapted from https://pytorch.org/vision/stable/auto_examples/others/plot_visualization_utils.html#sphx-glr-auto-examples-others-plot-visualization-utils-py

    :param input_images: List of images of type torch.Tensor
    :param max_cols: Maximum number of columns per row. The rest will be wrapped into the next row.
    """
    if not isinstance(input_images, list):
        input_images = [input_images]
    if not input_images:
        raise ValueError("No images provided to show, input_images cannot be empty")

    # Calculate the number of rows needed based on max_cols
    ncols = min(len(input_images), max_cols)  # Ensure no more than max_cols columns
    nrows = math.ceil(len(input_images) / ncols)  # Wrap images into multiple rows if necessary

    # Create the subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=(ncols * 5, nrows * 4))

    for i, img in enumerate(input_images):
        img = F.to_pil_image(img)
        
        # Calculate row and column index
        row = i // ncols
        col = i % ncols
        
        axs[row, col].imshow(np.asarray(img))
        axs[row, col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()

    # Use the right method to show the plot
    if is_running_in_docker():
        display(fig)  # To enforce inline plotting
        plt.show()
    else:
        plt.show()  # Normal display outside Docker
    
    plt.close(fig)  # Prevents duplicate figures in Jupyter