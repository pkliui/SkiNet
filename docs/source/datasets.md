# Datasets and dataloaders

This document describes the dataset system used in SkiNet.
See the [API reference](api/api_datasets) for full parameter documentation.

---

## Quickstart

- At present, only segmentation datasets are supported.
- The recommended way to create datasets is via `create_segmentation_datasets_from_config`, which handles
  splitting, transform construction, and dataset instantiation in one call.
- The resulting `DatasetSplit` container provides train, val, and test datasets that can be passed to
  PyTorch dataloaders directly:

```python
from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml
from SkiNet.ML.datasets.dataset_factory import create_segmentation_datasets_from_config
from torch.utils.data import DataLoader

cfg = load_config_from_yaml(cfg_path)
datasets = create_segmentation_datasets_from_config(cfg)

train_loader = DataLoader(datasets.train, batch_size=8, shuffle=True,  num_workers=4)
val_loader   = DataLoader(datasets.val,   batch_size=8, shuffle=False, num_workers=4)
```

- See {py:class}`SkiNet.ML.datasets.dataset_factory.DatasetFactory` for factory internals and extension.
- Alternatively, dataloaders can be created from `create_dataloaders_from_datasets`:

```python
from SkiNet.ML.dataloaders.create_dataloaders import create_dataloaders_from_datasets
from SkiNet.ML.datasets.dataset_factory import DatasetSplit, create_segmentation_datasets_from_config
from SkiNet.ML.datasets.segmentation_dataset import SegmentationDataset

segm_datasets: DatasetSplit[SegmentationDataset] = create_segmentation_datasets_from_config(main_config)
loaders = create_dataloaders_from_datasets(segm_datasets, main_config.trainconfig)
print("Train dataloader: ", loaders.train)
print("Val dataloader: ", loaders.val)
print("Test dataloader: ", loaders.test)
```

Under the hood this creates `RepeatDataLoader` instances that avoid spawning new worker processes each epoch
— see [Note about modification to PyTorch's default dataloader](dataloaders.md).

---

## Dataset Class

### `SegmentationDataset` (ISIC 2017)

`SkiNet.ML.datasets.segmentation_dataset.SegmentationDataset` implements a PyTorch `Dataset` for
semantic segmentation. Each sample is an image–mask pair. It is the only dataset the factory builds
(registered in `_DATASET_FACTORIES` for `ExperimentType.SEGMENTATION`).

Constructor signature:
```python
SegmentationDataset(
    data_root: Path,
    dataframe: pd.DataFrame,
    transform: SampleTransformAdapter,
    mode: MLWorkflowState,
    cache_in_ram: bool = True,
)
```

Each item returned by `__getitem__` is a dict:

```python
item  = dataset[0]
image = item["image"]   # transformed image tensor
mask  = item["mask"]    # transformed mask tensor
specs = item["specs"]   # sample metadata dict
```

`cache_in_ram=True` pre-loads all images into a RAM dict at construction time using a
`ThreadPoolExecutor` (up to 8 workers). Workers then only perform augmentation, never disk reads.
Set `cache_in_ram=False` for ISIC 2017 or other large datasets when RAM is limited.

---

## Dataset Splits

Splitting into train/val/test subsets is handled by `split_segmentation_metadata`,
which returns a `DataFrameSplits` object. The factory calls this internally; call it
directly if you need the raw DataFrames:

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

For ISIC 2017, `get_split_config()` returns a `SplitConfig` that uses the `predefined_split` column
instead of a random split, respecting the official challenge splits. Set
`predefined_split_column: null` in `DATA_CONFIG` to fall back to random splitting.

---

## Modifications to the default Dataset class

The following describes modifications to PyTorch's default Dataset class to prevent the
"Memory-on-copy" problem documented in
https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662

### Background and Motivation

Jupyter notebook: `SkiNet/ML/datasets/experiments/MemoryUsage_Dataset.ipynb`

When `num_workers > 0`, memory usage increases with each epoch for some users due to copy-on-write
behaviour in forked Python processes. From the GitHub thread:

> "If your DataLoaders iterate across a list of filenames, the references to that list add up over
> time, occupying memory. ... The simplest workaround is to replace native Python objects (dicts, lists)
> with array objects that only have one refcount (pandas, numpy, pyarrow)."

SkiNet stores file paths in `np.array(..., dtype=np.bytes_)` rather than Python lists.

---

## Augmentation of data

Augmentation uses [Albumentations](https://albumentations.ai). For training, the process has four stages:
1. Cropping to a pre-defined size
2. Spatial transformations (applied to both image and mask)
3. Photometric transformations (image only)
4. Normalization and conversion to tensor

### Transformation pipelines

Default pipelines for cropping, spatial, and photometric transforms are defined in
`SkiNet/ML/transformations/transform_pipelines.py`. By default, all transformations except
normalization are **disabled** and must be enabled in the YAML config.

Affine rotation uses nearest-neighbor interpolation (Albumentations defaults to `cv2.INTER_NEAREST`
for mask targets), preserving pixel values at 0 or 255 for uint8 masks.

Normalization mode is controlled by `normalization_mode` in `TRANSFORM_CONFIG`:

| Mode | Behaviour | When to use |
|---|---|---|
| `"image_per_channel"` (default) | Per-sample per-channel mean/std on the augmented image | General purpose; no dataset stats needed |
| `"standard"` | Fixed global mean/std from `normalization_mean` / `normalization_std` | When dataset statistics are pre-computed |

Example for ISIC 2017 pre-computed statistics:
```yaml
TRANSFORM_CONFIG:
  normalization_mode: "standard"
  normalization_mean: [0.699, 0.556, 0.512]
  normalization_std:  [0.158, 0.156, 0.171]
```

`normalization_mean` and `normalization_std` are required when `normalization_mode: "standard"` —
a `ValueError` is raised at pipeline construction time if they are absent.

### Configuration for transformation

```yaml
TRANSFORM_CONFIG:
  seed_value: 100
  crop:
    crop_apply: False
    crop_type: "random_resized_crop"
    size: [256, 256]
    scale: [0.8, 1.0]
  spatial_augmentation:
    square_symmetry_apply: True
    square_symmetry_p: 0.5
    affine_apply: True
    affine_scale: [0.8, 1.0]
    affine_translate_percent: {"x": [-0.05, 0.05], "y": [-0.05, 0.05]}
    affine_rotate: [-20, 20]
    perspective_apply: True
    perspective_scale: [0.05, 0.1]
    perspective_p: 0.2
    elastic_apply: True
    elastic_alpha: 120
    elastic_sigma: 10
    elastic_p: 0.1
  photometric_augmentation:
    color_jitter_apply: True
    color_jitter_brightness: 0.2
    color_jitter_contrast: 0.2
    color_jitter_p: 0.3
    gaussian_blur_apply: True
    gaussian_blur_sigma_limit: [0.5, 2.0]
    gaussian_blur_p: 0.2
    gaussian_noise_apply: True
    gaussian_noise_std_range: [0.01, 0.05]
    gaussian_noise_p: 0.2
```

The crop size must be divisible by `stride ^ number_of_downsampling_layers` (e.g., 16 for 4
downsampling layers with stride 2). This is validated at config-load time by
{py:class}`SkiNet.ML.configs.experiment_config.ExperimentConfig`.

### Seeding and reproducibility

- `seed_value` controls the augmentation sequence.
- **Using the same seed with different `num_workers` settings produces different augmentation sequences.**
  Reproducibility requires the same seed AND the same `num_workers`.
- Each worker gets a unique seed: `base_seed + torch.initial_seed()`.

### Train, val, and test pipelines

Training augmentations are strictly separate from validation and test preprocessing.
`get_transform_from_config` builds mode-specific pipelines:
- **train**: stochastic spatial and photometric augmentations
- **val/test**: deterministic steps only (crop, pad, normalize)

```python
from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml
from SkiNet.ML.transformations.transform_data import get_transform_from_config, TransformsContainer

cfg = load_config_from_yaml(cfg_path)
transform: TransformsContainer = get_transform_from_config(cfg)

print("train pipeline: ", transform.train)
print("val pipeline: ", transform.val)
print("test pipeline: ", transform.test)
```

### Visual inspection

```python
from SkiNet.ML.transformations.plot_transformed_data import visualize_augmented_data

dataset = SegmentationDataset(data_root=..., dataframe=..., transform=transform.train, mode=...)
visualize_augmented_data(dataset, samples=20)
```

Saves individual images and a grid of augmented samples to a specified directory.
