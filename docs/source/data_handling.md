# Data handling

Depending on data at hand, one may need to prepare them in this or that way prior to use in training. This page describes functions


## Image-mask pairing

- For segmentation, iamges and masks must be correctly paired to ensure the correctness of the results. Given full paths to images and masks, ```SkiNet.SkiNet.ML.utils.data_utils.filter_missing_images_and_masks```, for example, modifies the provided paths by including only those that have the same sample number both for image and mask

