# Plotting

Quick guide to plotting tasks in SkiNet.

Notebook with Docker examples: `SkiNet/Plotting/docker_plotting_examples.ipynb`

---

## Logging

Log skipped images to stdout and a file:

```python
import logging
from SkiNet.Utils.loggers import file_logging, stdout_logging
stdout_logging(logging.DEBUG)
file_logging()
```

---

## Plot images and masks side-by-side

### From a config-based dataset (ISIC 2017)

`plot_images_masks_side_by_side_matplotlib(dataset, num_samples=5)` indexes `dataset[i]` for
`i in range(num_samples)` and renders each image and mask with a colorbar. Pass any split from the
factory (`datasets.train`, `datasets.val`, `datasets.test`):

```python
from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml
from SkiNet.ML.datasets.dataset_factory import create_segmentation_datasets_from_config
from SkiNet.Plotting.plot_images_masks_side_by_side import plot_images_masks_side_by_side_matplotlib

cfg = load_config_from_yaml("main_config.yaml")
datasets = create_segmentation_datasets_from_config(cfg)
plot_images_masks_side_by_side_matplotlib(datasets.val, num_samples=3)
```

The transforms applied to each split are fixed by `get_transform_from_config` (train = augmented,
val/test = deterministic); the plotter renders whatever the dataset's transform produces.

---

## Plot images overlaid with masks

### From a folder using glob patterns

`folder` mode requires **both** `search_pattern_images` and `search_pattern_masks` (a missing or
`None` mask pattern raises `ValueError`). Patterns are passed to `Path(data_root).rglob(...)`, and
`filter_and_pair_valid_paths` drops unpaired files and pairs whose image/mask sizes differ. Point
`data_root` at the parent holding both the `*_Data` and `*_Part1_GroundTruth` directories:

```python
from SkiNet.Plotting.plot_segmentations import plot_segmentations

plot_segmentations(
    mode="folder",
    data_root="/mnt/data",
    search_pattern_images="ISIC-2017_*_Data/*/*.jpg",
    search_pattern_masks="ISIC-2017_*_Part1_GroundTruth/*/*_segmentation.png",
    max_cols=5,
    max_images_to_plot=10,
    alpha=0.5,
)
```

### Dataloader mode (retired)

> `plot_segmentations(mode="dataloader", ...)` is retired — it depended on the removed
> `DatasetSplitter` and on `DatasetClass(data_root=...)`, a signature only the legacy `PH2Dataset`
> supported, and now raises `NotImplementedError`. To overlay masks for ISIC 2017, build the dataset
> from config and use the side-by-side or augmented-data plotters below.

---

## Plot augmented data

```python
from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml
from SkiNet.ML.transformations.transform_data import get_transform_from_config
from SkiNet.ML.datasets.dataset_factory import create_segmentation_datasets_from_config
from SkiNet.ML.transformations.plot_transformed_data import visualize_augmented_data

cfg = load_config_from_yaml("main_config.yaml")
transform = get_transform_from_config(cfg)
datasets = create_segmentation_datasets_from_config(cfg)

visualize_augmented_data(dataset=datasets.train, samples=20)
```

Saves individual images and a grid of augmented samples to a specified directory.
