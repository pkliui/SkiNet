# Plotting

Quick guide how to do various plotting tasks


## Plot random images and masks for overview

- To quickly view  images and masks sitting in a certain folder, use ```SkiNet.Plotting.plot_random_samples```,
- where you have to specify the dataset's name e.g. PH2Dataset
- path to a folder where the data are located e.g. /local_folder/data/
- number of images to plot


```python

python plot_random_samples.py --dataset-name DATASET_NAME --path-to-data PATH_TO_DATA --num-images-to-plot NUM_IMAGES_TO_PLOT
```