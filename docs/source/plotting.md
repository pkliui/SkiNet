# Plotting

Quick guide how to do various plotting tasks
Notebook: (docker_plotting_examples.ipynb)[../SkiNet/Plotting/docker_plotting_examples.ipynb]


# Plot images and masks side-by-side

## Logging

Logging may be useful to track skipped images

```
import logging
from SkiNet.Utils.loggers import file_logging, stdout_logging
stdout_logging(logging.DEBUG)
file_logging()
```


## From a local directory and a specified dataset, as raw data
### When can be used: 
- Images and masks sitting in a local directory that can be used with a specific dataset

```
from SkiNet.Plotting.plot_images_masks_side_by_side import plot_images_masks_side_by_side_matplotlib
from SkiNet.ML.datasets.ph2dataset import PH2Dataset

dataset = PH2Dataset(data_root="/workplace/SkiNet/PH2_Dataset_images")
plot_images_masks_side_by_side_matplotlib(dataset, num_samples=3)
```


## From a local directory and a specified dataset + manually specified transformations
### When can be used: 
- Images and masks sitting in a local directory that can be used with a specific dataset
- Transformations provided as an albumentations.Compose object

Example:
```python
from SkiNet.Plotting.plot_images_masks_side_by_side import plot_images_masks_side_by_side_matplotlib
from SkiNet.ML.datasets.ph2dataset import PH2Dataset
import albumentations as A
from SkiNet.ML.transformations.transform_data import TransformData
import torch

# specify transforms
transform = A.Compose([A.CenterCrop(height=500, width=500),
                      A.ToTensorV2()])

dataset = PH2Dataset(data_root="/workplace/SkiNet/PH2_Dataset_images",
                     transform=transform)
plot_images_masks_side_by_side_matplotlib(dataset, num_samples=3)
```



## From a local directory and a specified dataset +  transformations using a YAML config
- Images and masks sitting in a local directory that can be used with a specific dataset
- Transformations provided in a YAML file and as a default configuration


```python
# import default config
from SkiNet.ML.configs import transformations_config
config = transformations_config.get_default_config()


# import yaml settings
from SkiNet.Utils.project_paths_tests import TRANSFORMATION_CONFIGS_YAML_PATH 
config.merge_from_file(TRANSFORMATION_CONFIGS_YAML_PATH) # override from YAML
config.freeze() #  to prevent further modification

# obtain the transform
from SkiNet.ML.transformations.transform_data import make_transform_from_config
transform = make_transform_from_config(
    config,
    augmentation_required=True)

from SkiNet.ML.datasets.ph2dataset import PH2Dataset
from SkiNet.Plotting.plot_images_masks_side_by_side import plot_images_masks_side_by_side_matplotlib
dataset = PH2Dataset(data_root="/workplace/SkiNet/PH2_Dataset_images",
                     transform=transform)
plot_images_masks_side_by_side_matplotlib(dataset, num_samples=5)
```

# Plot images overlayed with masks

## From a local directory and a specified file structure
### When can be used: 
- Images and masks sitting in a local directory
- Known file format and file structure
- ```plot_segmentations``` ensures only masks and images of the same size and having a pair are plotted

Example given arguments:
```
# given arguments
from SkiNet.Plotting.plot_segmentations import plot_segmentations
plot_segmentations(mode = "folder",
                   data_root = "/workplace/SkiNet/PH2_Dataset_images",
                   search_pattern_images = "*_Dermoscopic_Image/*.bmp",
                   search_pattern_masks = "*_lesion/*.bmp",
                   max_cols = 5,
                   max_images_to_plot = 10)
```


## From a dataloader and a specified dataset
### When can be used: 
- Images and masks yielded by a dataloader and a specific dataset
- The dataset ensures it has only masks and images of the same size and they have a pair

 
Example given arguments:
```
# given arguments
from SkiNet.Plotting.plot_segmentations import plot_segmentations

plot_segmentations(mode = "dataloader",
                   data_root = "/workplace/SkiNet/PH2_Dataset_images",
                   dataset_class_name = "PH2Dataset",
                   max_cols = 2,
                   batch_size = 2,
                   max_batches_to_plot  = 1,
                   default_transform_visualisation=True,
                   alpha=0.5)
```