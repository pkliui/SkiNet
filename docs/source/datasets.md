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



