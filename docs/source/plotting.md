# Plotting

Quick guide how to do various plotting tasks


# Read and plot images and masks from a local directory

### When can be used: 
- Images and masks sitting in a local directory
- Known file format and file structure


Example:

```
from SkiNet.Plotting.get_data.get_images_and_masks import read_images_from_directory
from SkiNet.Plotting.plot_masks_over_images import plot_masks_over_images


# specify path to data and search patterns as well the number of images to plot
directory_path = '/workplace/SkiNet/PH2_Dataset_images'
search_pattern_images = '*_Dermoscopic_Image/*.bmp'
search_pattern_masks =  '*_lesion/*.bmp'
NUM_IMAGES = 10

# get the images and masks into a list by reading from the specified directory
images = read_images_from_directory(directory_path, search_pattern_images, max_num_images_to_return=NUM_IMAGES)
masks = read_images_from_directory(directory_path,search_pattern_masks, max_num_images_to_return=NUM_IMAGES)

# plot masks over images
plot_masks_over_images(images, masks, alpha=0.4, colors="white", max_cols=5)
```



# Plot images and masks using a config file


## Dataloader mode

### When can be used: 
- Visualise images and masks yielded by a specific dataset
- Specify json config file like this:
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

- Do the following imports and call "plot_segmentations" function using the config file
- 
```
import json
import os
from SkiNet.Plotting.plot_segmentations import plot_segmentations
from SkiNet.Utils.get_configs import get_config_from_yaml


config_path = "/workplace/SkiNet/SkiNet/ML/configs/ph2dataset_plotting_config.json"
plot_segmentations(mode = "dataloader",
                   config=get_config_from_yaml(config_path))
```

## Folder mode

### When can be used: 
- Can also be used to plot from a local folder for a specific type of file
- Specify json config file like this:
```
{
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
import json
import os
from SkiNet.Plotting.plot_segmentations import plot_segmentations
from SkiNet.Utils.get_configs import get_config_from_yaml

config_path = "/workplace/SkiNet/SkiNet/ML/configs/ph2dataset_plotting_config.json"
plot_segmentations(mode = "folder",
                   config=get_config_from_yaml(config_path))
```

# Plot random images and masks for overview

- To quickly view  images and masks sitting in a certain folder, use ```SkiNet.Plotting.plot_random_samples```,
- where you have to specify the dataset's name e.g. PH2Dataset
- path to a folder where the data are located e.g. /local_folder/data/
- number of images to plot


```python

python plot_random_samples.py --dataset-name DATASET_NAME --path-to-data PATH_TO_DATA --num-images-to-plot NUM_IMAGES_TO_PLOT
```

This will start a new Flask application and create a new dataset using its respective dataset class, e.g. ```PH2Dataset" for PH2 images. 
