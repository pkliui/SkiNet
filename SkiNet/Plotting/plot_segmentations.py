from typing import List, Optional, Tuple, Union

import PIL.Image
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

from SkiNet.Plotting.get_data.get_images_and_masks import \
    read_images_and_masks_from_directory
from SkiNet.Plotting.plot_masks_over_images import plot_masks_over_images
from SkiNet.Utils.dev_utils import is_running_in_docker

# to be able to plot in jupyter notebook in Docker
if is_running_in_docker():
    import matplotlib
    matplotlib.use("Agg")


class PlotSegmentations:
    """
    Class to plot images and their corresponding segmentations (masks) over each other.
    This class is designed to be used in two modes:
    1. From a DataLoader, where it will extract images and masks from the batches.
    2. From a directory, where it will read images and masks from the specified paths.

    Used in conjunction with the `plot_masks_over_images` function to visualize the results.
    """
    def __init__(self,
                 images: List[PIL.Image.Image],
                 masks: List[PIL.Image.Image],
                 alpha: float,
                 colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]],
                 max_cols: int):
        """
        :param images:
            A list of PIL images, each of shape (3, H, W) where 3 is the number of channels,
            and H, W are the height and width of the images
        :param masks:
            A list of PIL images, each of shape (H, W) or (num_masks, H, W) where num_masks is the number of masks,
            and H, W are the height and width of the masks
        :param alpha:
            Transparency level of the masks. A value between 0 (fully transparent) and 1 (fully opaque).
            Default is 0.3.
        :param colors:
            Colors for the masks. Can be a single color (e.g., "red" or (255, 0, 0)) or a list of colors for multiple masks.
            Default is "white".
        :param max_cols:
            Maximum number of columns in the grid. Images will wrap to the next row if this limit is exceeded.
            Default is 2.
        """
        self.images = images
        self.masks = masks
        self.alpha = alpha
        self.colors = colors
        self.max_cols = max_cols

    @classmethod
    def from_dataloader(cls,
                        dataloader: DataLoader,
                        max_batches_to_display: Optional[int] = None,
                        alpha: float = 0.3,
                        colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = "white",
                        max_cols: int = 2) -> "PlotSegmentations":
        """
        Creates an instance of the class using images and masks from a DataLoader.

        :param dataloader: A PyTorch DataLoader that provides batches of data.
        :param max_batches_to_display: The maximum number of batches to display.
            If None, all batches will be processed. Defaults to None.

        :return cls: An instance of the class initialized with images and masks extracted from the DataLoader.
        """

        all_images = []
        all_masks = []

        for batch_idx, dataitem in enumerate(dataloader):
            all_images.append(dataitem["image"])
            all_masks.append(dataitem["mask"])

            # Stop if the maximum number of batches to display is reached
            if max_batches_to_display is not None and (batch_idx + 1) >= max_batches_to_display:
                break

        # Concatenate all batches into a single tensor
        images_tensor: torch.Tensor = torch.cat(all_images, dim=0)
        masks_tensor: torch.Tensor = torch.cat(all_masks, dim=0)

        # convert to PIL as required by plot_masks_over_images
        pil_images = [ToPILImage()(img) for img in images_tensor]
        pil_masks = [ToPILImage()(mask) for mask in masks_tensor]

        return cls(pil_images, pil_masks, alpha=alpha, colors=colors, max_cols=max_cols)

    @classmethod
    def from_paths(cls,
                   images: List[PIL.Image.Image],
                   masks: List[PIL.Image.Image],
                   alpha: float = 0.3,
                   colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = "white",
                   max_cols: int = 2) -> "PlotSegmentations":
        return cls(images, masks, alpha=alpha, colors=colors, max_cols=max_cols)

    def __call__(self) -> None:
        plot_masks_over_images(self.images, self.masks, alpha=self.alpha, colors=self.colors, max_cols=self.max_cols)


def plot_segmentations(mode: str,
                       data_root: Optional[str] = None,
                       dataset_class_name: Optional[str] = None,
                       max_cols: Optional[int] = None,
                       batch_size: Optional[int] = None,
                       max_batches_to_plot: Optional[int] = None,
                       search_pattern_images: Optional[str] = None,
                       search_pattern_masks: Optional[str] = None,
                       max_images_to_plot: Optional[int] = None,
                       split: Optional[List[float]] = None,
                       seed: Optional[int] = 42,
                       split_type_to_plot: Optional[str] = "train",
                       alpha: Optional[float] = 0.3,
                       colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = "white",
                       **dataset_kwargs: object) -> None:
    """
    The main function to plot images and their corresponding segmentations (masks) over each other using PlotSegmentations class
    Two modes of operation are supported:
    1. From a DataLoader, where it will extract images and masks from the batches.
    2. From a directory, where it will read images and masks from the specified paths.


    :param mode: The mode of operation. Can be "dataloader" or "folder".
    :param data_root: Root directory for the data (required for "dataloader"  and for "folder" mode).

    :param dataset_class_name: Name of the dataset class (required for "dataloader" mode).
    :param max_cols: Maximum number of columns in the grid (required for "dataloader" mode).
    :param batch_size: Batch size for the dataloader (required for "dataloader" mode).
    :param max_batches_to_plot: Maximum number of batches to plot (required for "dataloader" mode).

    :param search_pattern_images: Search pattern for images (required for "folder" mode).
    :param search_pattern_masks: Search pattern for masks (required for "folder" mode).
    :param max_images_to_plot: Maximum number of images to plot (optional for "folder" mode).

    :param split: List of dataset split ratios. Default is [0.8, 0.1, 0.1] for train, validation, and test splits.
    :param seed: Random seed for dataset splitting. Default is 42
    :param split_type_to_plot: Dataset split type to plot (e.g., "train", "val", "test") . Default is = "train",
    :param alpha: Transparency level of the masks. A value between 0 (fully transparent) and 1 (fully opaque). Default is 0.3.
    :param colors: Colors for the masks. Can be a single color (e.g., "red" or (255, 0, 0)) or a list of colors for multiple masks. Default is "white".


    Example to plot from a folder (ISIC 2017; data_root holds both the *_Data and
    *_Part1_GroundTruth directories):
    --------------------------------

    ```
    from SkiNet.Plotting.plot_segmentations import plot_segmentations
    plot_segmentations(mode = "folder",
                    data_root = "/mnt/data",
                    search_pattern_images = "ISIC-2017_*_Data/*/*.jpg",
                    search_pattern_masks = "ISIC-2017_*_Part1_GroundTruth/*/*_segmentation.png",
                    max_cols = 5,
                    max_images_to_plot = 10)
    ```

    Note:
    -----
    The "dataloader" mode is retired. It relied on splitting an in-memory ``Dataset`` via the
    removed ``DatasetSplitter`` and on constructing datasets with ``DatasetClass(data_root=...)``,
    a signature only the legacy ``PH2Dataset`` supported. For config-driven (ISIC 2017) datasets,
    plot from the factory instead:

    ```
    from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml
    from SkiNet.ML.datasets.dataset_factory import create_segmentation_datasets_from_config
    from SkiNet.ML.transformations.plot_transformed_data import visualize_augmented_data

    datasets = create_segmentation_datasets_from_config(load_config_from_yaml("main_config.yaml"))
    visualize_augmented_data(dataset=datasets.train, samples=20)
    ```
    """

    if mode == "dataloader":
        raise NotImplementedError(
            "plot_segmentations(mode='dataloader') is retired: it depended on the removed "
            "DatasetSplitter and on DatasetClass(data_root=...), which only the legacy PH2Dataset "
            "supported. For ISIC 2017, build datasets via create_segmentation_datasets_from_config "
            "and plot with visualize_augmented_data or plot_images_masks_side_by_side_matplotlib. "
            "Use mode='folder' to plot raw image/mask files from disk."
        )

    elif mode == "folder":
        # Validate required parameters for folder mode
        if not data_root or not search_pattern_images or not search_pattern_masks:
            raise ValueError("Missing required parameters for 'folder' mode.")

        # read both images and masks - only those that have a pair and are of the same size
        images, masks = read_images_and_masks_from_directory(
            directory_path=data_root,
            search_pattern_images=search_pattern_images,
            search_pattern_masks=search_pattern_masks,
            max_num_images_to_return=max_images_to_plot if max_images_to_plot is not None else 1,
        )  # -> Tuple[List[Image.Image], List[Image.Image]]
        plotter = PlotSegmentations.from_paths(
            images,  # -> List[Image.Image]
            masks,
            alpha=alpha if alpha is not None else 0.3,
            colors=colors if colors is not None else "white",
            max_cols=max_cols if max_cols is not None else 2,
        )
        plotter()
    else:
        raise ValueError(f"Unknown plotting mode: {mode}. Should be 'dataloader' or 'folder'.")


def validate_plotting_config(config: dict, mode: str) -> None:
    """
    Validates the configuration dictionary for the given mode.

    :param config: The configuration dictionary to validate.
    :param mode: The mode of operation. Can be "dataloader" or "folder".
    :raises ValueError: If required keys are missing from the configuration.
    """
    if mode == "dataloader":
        required_keys = [
            "split", "seed", "batch_size", "split_type_to_plot", "dataset_class_name", "max_batches_to_plot"
        ]
        missing_keys = [key for key in required_keys if key not in config["dataloader"]]
        if missing_keys:
            raise ValueError(f"Missing dataloader config keys: {missing_keys}")

    elif mode == "folder":
        required_keys = ["search_pattern_images", "search_pattern_masks", "max_images_to_plot"]
        missing_keys = [key for key in required_keys if key not in config["folder"]]
        if missing_keys:
            raise ValueError(f"Missing folder config keys: {missing_keys}")
