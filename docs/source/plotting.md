# Plotting

Quick guide how to do various plotting tasks
Notebook: (docker_plotting_examples.ipynb)[../SkiNet/Plotting/docker_plotting_examples.ipynb]


# Plot images and masks side-by-side

## From a local directory and a specified dataset
### When can be used: 
- Images and masks sitting in a local directory that can be used with a specific dataset

Example:
```
from SkiNet.Plotting.plot_images_masks_side_by_side import plot_images_masks_side_by_side_matplotlib
from SkiNet.ML.datasets.ph2dataset import PH2Dataset

dataset = PH2Dataset(data_root="/workplace/SkiNet/PH2_Dataset_images")
plot_images_masks_side_by_side_matplotlib(dataset, num_samples=1)
```



# Plot images overlayed with masks

## From a local directory and a specified file structure
### When can be used: 
- Images and masks sitting in a local directory
- Known file format and file structure

Example given arguments:
```
from SkiNet.Plotting.plot_segmentations import plot_segmentations
plot_segmentations(mode = "folder",
                   data_root = "/workplace/SkiNet/PH2_Dataset_images",
                   search_pattern_images = "*_Dermoscopic_Image/*.bmp",
                   search_pattern_masks = "*_lesion/*.bmp",
                   max_cols = 5,
                   max_images_to_plot = 10)
```


Example given config file:
```
{
    "dataloader": {
      "dataset_class_name": "PH2Dataset",
      "split": [0.8, 0.1, 0.1],
      "seed": 42,
      "split_type_to_plot": "train",
      "batch_size": 5,
      "max_batches_to_plot": 2

    },
    "data_root": "/workplace/SkiNet/PH2_Dataset_images",
    "alpha": 0.3,
    "colors": "white",
    "max_cols": 5
  }
```
```
from SkiNet.Plotting.plot_segmentations import plot_segmentations
from SkiNet.Utils.get_configs import get_config_from_yaml

        
config_path = "/workplace/SkiNet/SkiNet/ML/configs/ph2dataset_plotting_config.json"
plot_segmentations(mode = "folder",
                   config=get_config_from_yaml(config_path))
```



## From a dataloader and a specified dataset
### When can be used: 
- Images and masks yielded by a dataloader and a specific dataset
 
Example given arguments:
```
from SkiNet.Plotting.plot_segmentations import plot_segmentations
plot_segmentations(mode = "dataloader",
                   data_root = "/workplace/SkiNet/PH2_Dataset_images",
                   dataset_class_name = "PH2Dataset",
                   max_cols = 5,
                   batch_size = 10,
                   max_batches_to_plot  = 2)
```

Example given config file:
```

{
    "dataloader": {
      "dataset_class_name": "PH2Dataset",
      "split": [0.8, 0.1, 0.1],
      "seed": 42,
      "split_type_to_plot": "train",
      "batch_size": 5,
      "max_batches_to_plot": 2

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
```
```
# given config file
from SkiNet.Plotting.plot_segmentations import plot_segmentations
from SkiNet.Utils.get_configs import get_config_from_yaml

        
config_path = "/workplace/SkiNet/SkiNet/ML/configs/ph2dataset_plotting_config.json"
plot_segmentations(mode = "dataloader",
                   config=get_config_from_yaml(config_path))
```