# Datasets

- This document describes Datasets used in SkiNet
  
## Modifications to the default Dataset class

The following describes a few modifications to PyTorch's default Dataset class used in SkiNet in order to prevent the so-called "Memory-on_copy" prolem that was observed
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

Data are augmented using ```TransformData``` class in ```SkiNet/ML/transformations/transform_data.py```. In practice one can make a transformation pipeline using ```make_transform_from_config``` function and a respective configuration object ```config```. 

### Configuration for transformation
- The configuration options for transformations are defined using (YASC library)[https://github.com/rbgirshick/yacs].
- ```SkiNet/ML/configs/transformations_config.py```is a project config file for performing transformations in SkiNet that holds the default settings.
- For each experiment one typically creates a dedicated YAML configuration file where the options that are different from the default options are listed. Example:

```yaml
augmentation:
  random_affine_apply: True
  random_affine:
    degrees: 90
    translate: (0.1, 0.1)

  random_rotation_apply: False

  crop_apply: True
  center_crop:
    size: (400, 400)
```

- The imported config settings are overriden by the YAML file's content and then frozen to prevent any further modifications:

```python
# import default config
from SkiNet.ML.configs import transformations_config
config = transformations_config.get_default_config()

# import yaml settings
from SkiNet.Utils.project_paths_tests import TRANSFORMATION_CONFIGS_YAML_PATH 
config.merge_from_file(TRANSFORMATION_CONFIGS_YAML_PATH) # override from YAML
config.freeze() #  to prevent further modification
```


### Making transformations

- This configuration object ```config``` can now be used to make a transformation pipeline using ```make_transform_from_config```:

```python
from SkiNet.ML.transformations.transform_data import make_transform_from_config

transform_from_config = make_transform_from_config(
    config,
    augmentation_required=True)
transformed_image = transform_from_config(input_image)
```

- Since class ```TransformData```uses```v2.transforms```under the hood, the input image can be a torch.Tensor, TVImage, or PIL.Image.Image or a tuple of these types.  According to the torch documentation, v2.transforms support arbitrary input structures, such as single image, a tuple or a dictionary. The same structure will be returned as output. Pure torch.Tensor objects are treated as images. However, if the input is an Image, Video, or PIL.Image.Image instance, all other pure tensors are passed-through (not transformed). If there is no Image or Video instance, only the first pure torch.Tensor will be transformed as image or video, while all others will be passed-through. Here “first” means “first in a depth-wise traversal”.
 https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py





