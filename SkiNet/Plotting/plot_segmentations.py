from typing import List, Optional, Union

import PIL
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage

from SkiNet.ML.dataloaders.dataloaders import RepeatDataLoader
from SkiNet.ML.datasets.dataset_splitter import DatasetSplitter
from SkiNet.ML.datasets.ph2dataset import PH2Dataset
from SkiNet.ML.utils.configs.dynamic_class_loader import DynamicClassLoader
from SkiNet.ML.utils.model_utils import MLWorkflowState, state_mapping
from SkiNet.Plotting.get_data.get_images_and_masks import \
    read_images_from_directory
from SkiNet.Plotting.plot_masks_over_images import plot_masks_over_images
from SkiNet.Utils.dev_utils import is_running_in_docker

# to be able to plot in jupyter notebook in Docker
if is_running_in_docker():
    import matplotlib
    matplotlib.use("Agg")

from IPython.display import display


class PlotSegmentations:
    """
    Class to plot images and their corresponding segmentations (masks) over each other.
    This class is designed to be used in two modes:
    1. From a DataLoader, where it will extract images and masks from the batches.
    2. From a directory, where it will read images and masks from the specified paths.

    Used in conjunction with the `plot_masks_over_images` function to visualize the results.
    """
    def __init__(self,
                 images: List[Union[PIL.Image.Image, torch.Tensor]],
                 masks: List[Union[PIL.Image.Image, torch.Tensor]],
                 alpha: float,
                 colors: str,
                 max_cols: int):
        """
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
                        **kwargs):
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
            print(dataitem["image"].shape)
            all_images.append(dataitem["image"])
            all_masks.append(dataitem["mask"])

            # Stop if the maximum number of batches to display is reached
            if max_batches_to_display is not None and (batch_idx + 1) >= max_batches_to_display:
                break

        # Concatenate all batches into a single tensor
        all_images = torch.cat(all_images, dim=0)
        print(all_images.shape)
        all_masks = torch.cat(all_masks, dim=0)

        return cls(all_images, all_masks, **kwargs)
    
    @classmethod
    def from_paths(cls,
                   images,
                   masks,
                   **kwargs):
        return cls(images, masks, **kwargs)

    def __call__(self):
        plot_masks_over_images(self.images, self.masks, alpha=self.alpha, colors=self.colors, max_cols=self.max_cols)


def plot_segmentations(mode: str,
                       config: Optional[dict] = None,
                       data_root: str = None,
                       dataset_class_name: str = None,
                       max_cols: int = None,
                       batch_size: int = None,
                       max_batches_to_plot: int = None,
                       search_pattern_images: str = None,
                       search_pattern_masks: str = None,
                       max_images_to_plot: int = None,
                       split: Optional[List[float]] = [0.8, 0.1, 0.1],
                       seed: Optional[int] = 42,
                       split_type_to_plot: Optional[str] = "train",
                       alpha: Optional[float] = 0.3,
                       colors: Optional[Union[str, List[str]]] = "white"):
    """
    The main function to plot images and their corresponding segmentations (masks) over each other
    It is based on the PlotSegmentations class and can be used either with a configuration file or with direct arguments to plot in the following modes:
    
    1. From a DataLoader, where it will extract images and masks from the batches.
    2. From a directory, where it will read images and masks from the specified paths.

    
    :param mode: The mode of operation. Can be "dataloader" or "folder".
    :param config: Optional configuration dictionary. Parameters in this dictionary will be overridden by direct arguments.
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


    Example config  JSON file:
    --------------------------------
    {
        "dataloader": {
        "dataset_class_name": "PH2Dataset",
        "max_batches_to_plot": 2,
        "split": [0.8, 0.1, 0.1],
        "seed": 42,
        "batch_size": 5,
        "num_workers": 1,
        "shuffle": false,
        "split_type_to_plot": "train"
        },
        "folder": {
        "search_pattern_images": "*_Dermoscopic_Image/*.bmp",
        "search_pattern_masks": "*_lesion/*.bmp",
        "max_images_to_plot": 10
        },
        "data_root": "/workplace/SkiNet/PH2_Dataset_images",
        "alpha": 0.3,
        "colors": "white",
        "max_cols": 5
    }

    Example plotting from a dataloader:
    --------------------------------
    from SkiNet.Plotting.plot_segmentations import plot_segmentations
    from SkiNet.Utils.get_configs import get_config_from_yaml
    config_path = "/workplace/SkiNet/SkiNet/ML/configs/ph2dataset_plotting_config.json"
    plot_segmentations(mode = "dataloader",
                    config=get_config_from_yaml(config_path))

    Example plotting from a folder:
    --------------------------------
    from SkiNet.Plotting.plot_segmentations import plot_segmentations
    from SkiNet.Utils.get_configs import get_config_from_yaml
    plot_segmentations(mode = "folder",
                    config=get_config_from_yaml(config_path))



                        
    Example using direct arguments
    --------------------------------
    When plotting from a DataLoader (mode="dataloader") using direct arguments, the following ones are required
    
    to plot from a dataloader:
    --------------------------------

    from SkiNet.Plotting.plot_segmentations import plot_segmentations
    plot_segmentations(mode = "dataloader",
                    data_root = "/workplace/SkiNet/PH2_Dataset_images",
                    dataset_class_name = "PH2Dataset",
                    max_cols = 5,
                    batch_size = 10,
                    max_batches_to_plot  = 2)
    
    to plot from a folder:
    --------------------------------
    from SkiNet.Plotting.plot_segmentations import plot_segmentations
    plot_segmentations(mode = "folder",
                    data_root = "/workplace/SkiNet/PH2_Dataset_images",
                    search_pattern_images = "*_Dermoscopic_Image/*.bmp",
                    search_pattern_masks = "*_lesion/*.bmp",
                    max_images_to_plot = 10)                   

    """

    # Merge config with direct arguments (direct arguments take precedence)
    config = config or {}
    data_root = data_root or config.get("data_root")
    alpha = alpha or config.get("alpha")
    colors = colors or config.get("colors")
    max_cols = max_cols or config.get("max_cols")

    if mode == "dataloader":
        dataloader_config = config.get("dataloader", {})
        dataset_class_name = dataset_class_name or dataloader_config.get("dataset_class_name")
        split = split or dataloader_config.get("split")
        seed = seed or dataloader_config.get("seed")
        split_type_to_plot = split_type_to_plot or dataloader_config.get("split_type_to_plot")
        batch_size = batch_size or dataloader_config.get("batch_size")
        max_batches_to_plot = max_batches_to_plot or dataloader_config.get("max_batches_to_plot")

        # Validate required parameters for dataloader mode
        if not all([dataset_class_name, split, seed, split_type_to_plot, batch_size]):
            raise ValueError("Missing required parameters for 'dataloader' mode.")

        # Dynamically load the dataset class
        class_loader = DynamicClassLoader(class_name=dataset_class_name)
        DatasetClass = class_loader.load_the_class()

        if DatasetClass is None:
            raise ValueError(f"Failed to load dataset class: {dataset_class_name}")

        # Initialize the dataset
        dataset = DatasetClass(data_root=data_root)

        # Split the dataset
        datasets = DatasetSplitter.get_split_datasets(
            dataset,
            *split,
            random_seed=seed
        )

        dataset = datasets[state_mapping(split_type_to_plot)].dataset
        dataloader = RepeatDataLoader(dataset,
                                      batch_size=batch_size,
                                      num_workers=1,
                                      shuffle=False)

        plotter = PlotSegmentations.from_dataloader(
            dataloader,
            max_batches_to_display=max_batches_to_plot,
            alpha=alpha,
            colors=colors,
            max_cols=max_cols
        )
        plotter()

    elif mode == "folder":
        folder_config = config.get("folder", {})
        search_pattern_images = search_pattern_images or folder_config.get("search_pattern_images")
        search_pattern_masks = search_pattern_masks or folder_config.get("search_pattern_masks")
        max_images_to_plot = max_images_to_plot or folder_config.get("max_images_to_plot")

        # Validate required parameters for folder mode
        if not all([search_pattern_images, search_pattern_masks]):
            raise ValueError("Missing required parameters for 'folder' mode.")

        images = read_images_from_directory(data_root, search_pattern_images, max_num_images_to_return=max_images_to_plot)
        masks = read_images_from_directory(data_root, search_pattern_masks, max_num_images_to_return=max_images_to_plot)
        plotter = PlotSegmentations.from_paths(
            images,
            masks,
            alpha=alpha,
            colors=colors,
            max_cols=max_cols
        )
        plotter()
    else:
        raise ValueError(f"Unknown plotting mode: {mode}. Should be 'dataloader' or 'folder'.")


def validate_plotting_config(config: dict, mode: str):
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