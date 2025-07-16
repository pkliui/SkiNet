# Datasets

- This document describes Datasets used in SkiNet
  
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

Data augmentation is performed using the ```TransformData``` class in ```transform_data.py```. The transformation pipeline is constructed by the ```make_transform_from_config``` function, which adds transformations from Albumentations library, based on conditional checks of a provided configuration object (```config```). The resulting pipeline can then be passed to the "transform" argument of a dataset.

### Configuration for transformation
- The configuration options for transformations are defined using (YASC library)[https://github.com/rbgirshick/yacs].
- ```SkiNet/ML/configs/transformations_config.py``` is the project’s default configuration file for transformations in SkiNet.
- For each experiment, a dedicated YAML configuration file is typically created, specifying only the options that differ from the defaults. For example:

```yaml
augmentation:
  horizontal_flip_apply: True
  horizontal_flip:
    p: 0.5
  vertical_flip_apply: True
  vertical_flip:
    p: 0.5
  affine_apply: True
  affine:
    rotate: (-90, 90)
    translate_percent: (0.1, 0.1)
    shear: (-20, 20)
  colorjitter_apply: True
  colorjitter:
    brightness: 0.1
    contrast: 0.1
    saturation: 0.1
    p: 0.5
  center_crop_apply: True
  center_crop:
    height: 400
    width: 400
```

- A config object is created by overriding the defaults from an experiment-specific YAML and freezing the modified config.
- The configuration object can now be used to make a transformation pipeline using ```make_transform_from_config```:

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
```

- The pipeline can now be queried to display the transformations using .pipeline argument:
```
transform.pipeline,
```
which prints out the following:

```
Compose([
  HorizontalFlip(p=0.5),
  VerticalFlip(p=0.5),
  Affine(p=0.5, balanced_scale=False, border_mode=0, fill=0.0, fill_mask=0.0, fit_output=False, interpolation=1, keep_ratio=False, mask_interpolation=0, rotate=(-90.0, 90.0), rotate_method='largest_box', scale={'x': (1.0, 1.0), 'y': (1.0, 1.0)}, shear={'x': (-20.0, 20.0), 'y': (-20.0, 20.0)}, translate_percent={'x': (0.1, 0.1), 'y': (0.1, 0.1)}, translate_px=None),
  ColorJitter(p=0.5, brightness=(0.9, 1.1), contrast=(0.9, 1.1), hue=(-0.5, 0.5), saturation=(0.9, 1.1)),
  CenterCrop(p=1.0, border_mode=0, fill=0.0, fill_mask=0.0, height=400, pad_if_needed=False, pad_position='center', width=400),
  ToTensorV2(p=1.0, transpose_mask=False),
], p=1.0, bbox_params=None, keypoint_params=None, additional_targets={}, is_check_shapes=True)
```

Note, "ToTensorV2(p=1.0, transpose_mask=False)" at the end.



### Transforming data in datasets

Datasets accept this transformation pipeline via "transform" argument:

```
from SkiNet.ML.datasets.ph2dataset import PH2Dataset
dataset = PH2Dataset(
  data_root="/workplace/SkiNet/PH2_Dataset_images",
  transform=transform)
```



