"""
Datasets used in SkiNet/ML/datasets/utils/MemoryUsage_Dalaloader_Ubuntu.ipynb 
in connection with "Copy-on-Write"  problem in Dataloaders

https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading

https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
"""

import io
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
# import AzureMachineLearningFileSystem here lazily
from azureml.fsspec import AzureMachineLearningFileSystem
from PIL import Image
from skimage.io import imread
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.io import read_image

from SkiNet.ML.utils.data_utils import (extract_sample_number,
                                        filter_missing_images_and_masks)


def get_ham_data_paths(data_root: Union[str, Path, AzureMachineLearningFileSystem], azure_data=False) -> Tuple[List[Path], List[Path]]:
    """
    Get paths to images and masks of HAM dataset.
    
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
    
    """

    if isinstance(data_root, (str, Path)):
        data_path = Path(data_root)
        if not data_path.exists():
            raise ValueError(f"The specified data_root does not exist: {data_root}")
    elif azure_data and isinstance(data_root, AzureMachineLearningFileSystem):
        # If it's an AzureMachineLearningFileSystem, assume it's valid.
        data_path = data_root
    else:
        raise ValueError("Provided data_root must be either a string, a Path, or an AzureMachineLearningFileSystem instance.")

    # Find all image and mask paths
    if azure_data:
        image_paths = list(data_root.glob('**.jpg'))
        mask_paths = list(data_root.glob('**.jpg'))
    else:
        image_paths = list(Path(data_root).rglob('*.jpg'))
        mask_paths = list(Path(data_root).rglob('*.jpg'))
    return image_paths, mask_paths




class HAMDatasetNpTensorDict(Dataset):
    """
    HAM Dataset class used for experimenting with the copy-on-write behaviour.
    
    The paths to images and masks were converted to numpy arrays of type bytes.
    The class returns a dictionary of tensors of images and masks
    """

    def __init__(self,
                 data_root: Union[str, Path, AzureMachineLearningFileSystem],
                 filter_inexisting_samples: Optional[bool] = True):
        """
        Initialize HAMDataset class

        :param data_root: if provided as a string, it is a local directory that contains folders with samples of data uniquely identifiable by their ID
                          otherwise it is an Azure Machine Learning filesystem instance referencing a specific location on Azure data storage
        :param filter_inexisting_samples: If True, includes only those paths to images and masks, in which all items are uniquely pairable by their sample number. Default is True
        """
        self.data_root = data_root
        if isinstance(self.data_root, (str, Path)):
            self.azure_data = False
        elif isinstance(self.data_root, AzureMachineLearningFileSystem):
            self.azure_data = True
        else:
            raise ValueError("Provided data_root must be either a string or a Path pointing to a local directory with data \
            or an AzureMachineLearningFileSystem instance pointing to data in Azure")
                
        self.images_list, self.masks_list = get_ham_data_paths(self.data_root, azure_data=self.azure_data)

        # convert lists to numpy arrays according to https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        self.images_list = np.array(self.images_list, dtype=np.bytes_)
        self.masks_list = np.array(self.masks_list, dtype=np.bytes_)
        """Numpy arrays of dtype np.bytes of full paths to images and masks"""
        

    def __getitem__(self, item):
        """
        Return a dictionary containing an image and a mask as tensors (normal practice)

        :param item: The index or key used to access a specific sample of data from the dataset.
        :return sample: A dictionary with keys 'image' and 'mask' containing a tensor image and a tensor mask sample, respectively
        """
        if self.azure_data:
            with self.data_root.open(self.images_list[item]) as img:
                image = Image.open(img)
            with self.data_root.open(self.masks_list[item]) as msk:
                mask = Image.open(msk)
        else:
            image = Image.open(self.images_list[item])
            mask = Image.open(self.masks_list[item])


        # Resize the image and mask (for example, to 256x256)
        resize_transform = transforms.Resize((1, 1)) # here just 1x1 to save memory
        image = resize_transform(image)
        mask = resize_transform(mask)

        transform = transforms.ToTensor()
        image = transform(image) 
        mask = transform(mask)

        return {'image': image, 'mask': mask}
            
    def __len__(self):
        # we assume filter_missing_images_and_masks is True whilst returning __len__(self) value, i.e. when the number of images and masks is the same aafter filter_missing_images_and_masks
        return len(self.images_list)




class HAMDataset(Dataset):
    """
    HAM Dataset class used for experimenting with the copy-on-write behaviour.
    
    The paths to images and masks are just lists (should lead to an increased memory usage and it does)
    The class returns a dictionary of tensors of images and masks
    """

    def __init__(self,
                 data_root: Union[str, Path, AzureMachineLearningFileSystem],
                 filter_inexisting_samples: Optional[bool] = True):
        """
        Initialize HAMDataset class

        :param data_root: if provided as a string, it is a local directory that contains folders with samples of data uniquely identifiable by their ID
                          otherwise it is an Azure Machine Learning filesystem instance referencing a specific location on Azure data storage
        :param filter_inexisting_samples: If True, includes only those paths to images and masks, in which all items are uniquely pairable by their sample number. Default is True
        """
        self.data_root = data_root
        if isinstance(self.data_root, (str, Path)):
            self.azure_data = False
        elif isinstance(self.data_root, AzureMachineLearningFileSystem):
            self.azure_data = True
        else:
            raise ValueError("Provided data_root must be either a string or a Path pointing to a local directory with data \
            or an AzureMachineLearningFileSystem instance pointing to data in Azure")
                
        self.images_list, self.masks_list = get_ham_data_paths(self.data_root, azure_data=self.azure_data)
        """A tuple of lists of full paths to images and masks"""
        

    def __getitem__(self, item):
        """
        Return a dictionary containing an image and a mask

        :param item: The index or key used to access a specific sample of data from the dataset.
        :return sample: A dictionary with keys 'image' and 'mask' containing a tensor image and a tensor mask sample, respectively
        """
        if self.azure_data:
            with self.data_root.open(self.images_list[item]) as img:
                image = Image.open(img)
            with self.data_root.open(self.masks_list[item]) as msk:
                mask = Image.open(msk)
        else:
            image = Image.open(self.images_list[item])
            mask = Image.open(self.masks_list[item])


        # Resize the image and mask (for example, to 256x256)
        resize_transform = transforms.Resize((1, 1))
        image = resize_transform(image)
        mask = resize_transform(mask)

        transform = transforms.ToTensor()
        image = transform(image) examples
        mask = transform(mask)

        return {'image': image, 'mask': mask}
            
    def __len__(self):
        # we assume filter_missing_images_and_masks is True whilst returning __len__(self) value, i.e. when the number of images and masks is the same aafter filter_missing_images_and_masks
        return len(self.images_list)

