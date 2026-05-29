# Datasets and dataloaders

This document describes the dataset system used in SkiNet.
See the [API reference](api/api_datasets) for full parameter documentation.

---

## Quickstart
- At present, only datasets for segmentation are supported
- The recommended way to create datasets is via a respective factory by calling ```create_segmentation_datasets_from_config``` function, which handles
splitting, transform construction, and dataset instantiation in one call
- The resulting container provides users with train, val and test datasets that can be passed to PyTorch dataloaders directly:

```python
from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml
from SkiNet.ML.datasets.dataset_factory import create_segmentation_datasets_from_config
from torch.utils.data import DataLoader

cfg = load_config_from_yaml(cfg_path)
train_val_test_dataset = create_segmentation_datasets_from_config(cfg)

train_loader = DataLoader(bundle.train, batch_size=8, shuffle=True,  num_workers=4)
val_loader   = DataLoader(bundle.val,   batch_size=8, shuffle=False, num_workers=4)
```

- See {py:class}`SkiNet.ML.datasets.dataset_factory.DatasetFactory` for factory internals and how to extend it.
- Alternatively, dataloaders can be created from ```SkiNet.ML.dataloaders.create_dataloaders.create_dataloaders_from_datasets``` function:

```python
from SkiNet.ML.dataloaders.create_dataloaders import create_dataloaders_from_datasets
from SkiNet.ML.datasets.dataset_factory import DatasetSplit, create_segmentation_datasets_from_config
from SkiNet.ML.datasets.segmentation_dataset import SegmentationDataset

# create segmentation dataset (train, val, test container) from config
segm_datasets: DatasetSplit[SegmentationDataset] = create_segmentation_datasets_from_config(main_config)
# create train, val, test dataloaders from it
loaders = create_dataloaders_from_datasets(segm_datasets, main_config.trainconfig)
print("Train dataloader: ", loaders.train)
print("Val dataloader: ", loaders.val)
print("Test dataloader: ", loaders.test)
```
- Under the hood, it creates train, val and test RepeatDataloaders that avoid spawning a new process after each epoch  - see [Note about modification to PyTorch's default dataloader](dataloaders.md)

---

## Dataset Classes

`SkiNet.ML.datasets.segmentation_dataset` implements a PyTorch `Dataset` for
**semantic segmentation**. Each sample is a pair:

- **image**: input raster (grayscale or RGB)
- **mask**: target binary raster

Each item returned by the dataset is a dict:

```python
item  = dataset[0]
image = item["image"]   # transformed image tensor
mask  = item["mask"]    # transformed mask tensor
specs = item["specs"]   # sample metadata dict
```

---

## Dataset Splits

Splitting into train/val/test subsets is handled by `split_segmentation_metadata`,
which returns a `DataFrameSplits` object. In normal usage this is called internally
by the factory, but it can be called directly if needed:

```python
from SkiNet.Utils.data.split_data import split_segmentation_metadata

main_config  = load_config_from_yaml(cfg_path)
split_config = main_config.dataconfig.get_split_config()
metadata_df  = main_config.dataconfig.metadata

splits   = split_segmentation_metadata(df=metadata_df, split_config=split_config)
train_df = splits.train
val_df   = splits.val
test_df  = splits.test
```

---

## Modifications to the default Dataset class

The following describes a few modifications to PyTorch's default Dataset class used in SkiNet in order to prevent the so-called "Memory-on_copy" problem that was observed
- in https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662

### Background and Motivation

Jupyter notebook is here: (MemoryUsage_Dataset.ipynb)[SkiNet/ML/datasets/experiments/MemoryUsage_Dataset.ipynb]

Whilst employing num_workers>0 in DalaLoader, the memory usage is increasing with each epoch for some users.

- There is a warning in Pytorch documentation: https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading with a reference to an issue on Github: https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662


Citing the commentator on Github, "If your Dataloaders iterate across a list of filenames, the references to that list add up over time, occupying memory. Strictly speaking this is not a memory leak, but a copy-on-access problem of forked python processes due to changing refcounts. It isn't a Pytorch issue either, but simply is due to how Python is structured.

[..]

```The simplest workaround is to replace native Python objects (dicts, lists) with array objects that only have one refcount (pandas, numpy, pyarrow)... You can also consider using torch tensors as torch tensor objects do not have copy-on-write behaviour. If you are storing strings, refer to these three comments on setting the numpy datatype correctly.```

A few other commenters have also shared workarounds using custom implementations: custom tensor-backed string array and custom StringArray and DictArray classes

Given that interactions with Python multiprocessing lies at the heart of this issue, a few commenters have replaced their Python objects using multiprocessing Manager, which handles shared states at here and here.

A few other notable comments:

shuffle=True exacerbates the memory issue
Workaround to increase shared memory
Workaround to increase number of allowed file descriptors
Add torch.cuda.empty_cache() at end of each iteration
Workaround by setting num_workers=0, but training will be slow"

Example jupyter notebook showing this problem (from the authors on Github) https://gist.github.com/mprostock/2850f3cd465155689052f0fa3a177a50

## Augmentation of data

Augmentation of data is done using (Albumenations library)[https://albumentations.ai]. For training, the process is split into four stages such as:
- cropping dataset to a pre-defined size
- spatial transformations (applied both to image and mask)
- photometric tranformations (only image)
- normalisation and convertion to tensor
At present, augmentation is applied uniformly across all samples with an option to assign a specific probability to each augmentation operation.

### Transformation pipelines

- Default transformation pipelines for cropping, spatial and photometric transformed are specified in ```SkiNet/ML/transformations/transform_pipelines.py``` as optional sequences of transformations, each applied with a certain probability. Every time the data loader serves an image, the pipeline generates a fresh random variant.
- By default, all transformations, except the noramlisation, are DISABLED and must be turned ON in the main YAML config

- As per default Albumentations settings, affine rotation uses nearest neighbor interpolation (Albumentations defaults to cv2.INTER_NEAREST for targets passed via the mask argument). The resulting pixel values are therefore preserved
and are 0 or 255 for uint8 images.
- Normalisation mode is controlled by `normalization_mode` in `TRANSFORM_CONFIG`:

  | Mode | Behaviour | When to use |
  |---|---|---|
  | `"image_per_channel"` (default) | Per-sample per-channel mean/std computed on the augmented image | General purpose; no dataset stats needed |
  | `"standard"` | Fixed global mean/std supplied via `normalization_mean` / `normalization_std` | When dataset statistics are pre-computed (e.g. ISIC 2017) |

  Example for `image_per_channel` (default, no extra fields needed):
  ```yaml
  TRANSFORM_CONFIG:
    normalization_mode: "image_per_channel"
  ```

  Example for `standard` with pre-computed ISIC 2017 statistics:
  ```yaml
  TRANSFORM_CONFIG:
    normalization_mode: "standard"
    normalization_mean: [0.699, 0.556, 0.512]
    normalization_std:  [0.158, 0.156, 0.171]
  ```

  `normalization_mean` and `normalization_std` are required when `normalization_mode: "standard"` — a `ValueError` is raised at pipeline construction time if they are absent.

- ```seed```can be used to ensure reproducibility.
- **Critical Note: Using the same seed with different num_workers settings will produce different augmentation sequences**. This is by design to ensure:
    - Each worker produces unique augmentations (no duplicates)
    - Maximum augmentation diversity in parallel processing
    - Reproducibility when using the SAME num_workers configuration
- **Key insight:**  The augmentation sequence depends on BOTH the seed AND num_workers. To get identical results, you must use the same seed AND the same num_workers.
- Each worker gets a unique, reproducible seed based on base_seed + torch.initial_seed()
- Seeds are automatically updated on worker respawn

### Configuration for transformation
- The configuration options for transformations are defined in the main configuration YAML file. If some of the fields required by transformation are missing,
  the default values from relevant cropping and augmentation classes are used.
- Users are expected to create their own configuration in the main config file. For example:
```yaml
TRANSFORM_CONFIG:
  crop:
    crop_apply: True
    crop_type: "random_resized_crop"
    size: [512, 512]
    scale: [0.8, 1.0]
  spatial_augmentation:
    affine_apply: True
    affine_scale: [0.5, 1.0]
    affine_translate_percent: {"x": [-0.05, 0.05], "y": [-0.05, 0.05]}
    affine_rotate: [-45, 45]
    affine_shear: {"x": [-15, 15], "y": [-15, 15]}
  photometric_augmentation:
    color_jitter_apply: True
    color_jitter_brightness: 0.2
    color_jitter_contrast: 0.2
    color_jitter_p: 0.5
```

- The crop size is expected to be a multiple of the number of downsampling layers times the downsampling stride
(e.g. for a model with 4 downsampling layers and stride=2 downsampling, this factor is 16)
- At present, the requirement is enforced on the {py:class}`SkiNet.ML.configs.experiment_config.ExperimentConfig`
 level

### Train, val and test pipelines
- The pipelines defined above provide a cenralized source of transformations in each of the stages (crop, spatial, photometric and normalisation)
- Training augmentations are kept strictly separate from validation and test preprocessing.
- For each execution mode (train/val/test),  ```TransformsContainer```  stores the pipeline that applies to that mode.
- ```get_transform_from_config``` builds these mode-specific pipelines: the training pipeline can include stochastic spatial and photometric augmentations,
while the validation and test pipelines use only deterministic steps such as resize, padding, and normalization (no geometric or photometric transformations):

**Example**
```python
from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml
from SkiNet.ML.transformations.transform_data import get_transform_from_config, TransformsContainer


# load config
cfg = load_config_from_yaml(cfg_path)
# load transforms
transform: TransformsContainer = get_transform_from_config(cfg)

print("train pipeline: ", transform.train)
print("val pipeline: ", transform.val)
print("test pipeline: ", transform.test)

# make a dataset
dataset = SegmentationDataset(config=cfg, transform=transform)
```
- This construct prevents augmentation from leaking into validation and test  modes.
- The transformation pipeline is constructed by the ```get_transform_from_config``` function,
which and can then be passed to the "transform" argument of a dataset:

### Verify the pipeline by visual inspection
To plot augmented samples use

```python
from SkiNet.ML.transformations.plot_transformed_data import visualize_augmented_data
visualize_augmented_data(dataset=dataset, samples=20)
```
- This will save individual images and a grid of augmented samples into a specified directory (can be disabled in arguments)
