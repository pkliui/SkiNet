# Plotting

Quick guide how to dod various plotting tasks


## Plot random images and masks for overview

- Edit plot_random_samples.py with

num_images = 4
ph2_dataset = PH2Dataset(
    root_dir=Path("/Users/Pavel/Documents/repos_data/UNet/PH2_Dataset_images/PH22")
)

- For quick overview of images and masks, run the following from terminal


```python
python SkiNet/Plotting/plot_random_samples.py
```