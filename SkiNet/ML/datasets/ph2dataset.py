import logging
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from SkiNet.Utils.image_utils import ensure_np_image
import albumentations as A
from azureml.fsspec import AzureMachineLearningFileSystem
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data.dataset import Dataset


# import AzureMachineLearningFileSystem here lazily
from SkiNet.ML.transformations.transform_data import TransformData
from SkiNet.ML.utils.data_utils import filter_and_pair_valid_paths


def get_ph2_data_paths(data_root: Union[str, Path, AzureMachineLearningFileSystem], 
                       azure_data: bool
                       ) -> Tuple[NDArray[np.bytes_], NDArray[np.bytes_]]:
    """
    Get paths to images and masks of a PH2 dataset. Paths to images and masks that do not have the same dimensions are filtered out.

    :param data_root: if provided as a string or Path, it is a local directory that contains folders with samples of data uniquely identifiable by their ID,
        otherwise it is an Azure Machine Learning filesystem instance referencing a specific location on Azure data storage. It is provided in PH2Dataset.__init__. 
    :param azure_data: If True, provided data_root points to a AzureMachineLearningFileSystem instance, otherwise if the data are stored locally.
        It is provided in PH2Dataset.__init__. 
    :return: A tuple of two numpy arrays of dtype np.bytes_ that contains full paths to images and masks. 
        This data type is required by the Dataset class to fix the copy-on-write problem that results in an increased memory usage 
        https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading

        
    PH2 dataset copyright: Teresa Mendonça, Pedro M. Ferreira, Jorge Marques, Andre R. S. Marcal, Jorge Rozeira.
    PH² - A dermoscopic image database for research and benchmarking,
    35th International Conference of the IEEE Engineering in Medicine and Biology Society, July 3-7, 2013, Osaka, Japan.

    The file structure is expected to be as following:
        * root_dir
            * sample1
                * sample1_Dermoscopic_Image
                    * sample1.bmp
                * sample1_lesion
                *    sample1_lesion.bmp
            * sample2
                * sample2_Dermoscopic_Image
                    * sample1.bmp
                * sample2_lesion
                    * sample2_lesion.bmp

    Note that only ONE image per folder is expected, e.g. one image in sample1_Dermoscopic_Image folder,
    one mask in sample1_lesion folder
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
        image_paths = list(data_root.glob('**_Dermoscopic_Image/**.bmp'))
        mask_paths = list(data_root.glob('**_lesion/**.bmp'))
    else:
        image_paths = list(Path(data_root).rglob('*_Dermoscopic_Image/*.bmp'))
        mask_paths = list(Path(data_root).rglob('*_lesion/*.bmp'))

    # filter missing pairs only, the size will be adjusted later
    image_paths, mask_paths = filter_and_pair_valid_paths(image_paths, mask_paths, False)

    # Convert lists to NumPy arrays of dtype np.bytes_ to fix copy-on-write problem that results in an increased memory usage
    # https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
    image_paths = np.array(image_paths, dtype=np.bytes_)
    mask_paths = np.array(mask_paths, dtype=np.bytes_)

    return image_paths, mask_paths


class PH2Dataset(Dataset):
    """
    Class for PH2 data

    Copyright: Teresa Mendonça, Pedro M. Ferreira, Jorge Marques, Andre R. S. Marcal, Jorge Rozeira.
    PH² - A dermoscopic image database for research and benchmarking,
    35th International Conference of the IEEE Engineering in Medicine and Biology Society, July 3-7, 2013, Osaka, Japan.
    """
    def __init__(self,
                 data_root: Union[str, Path, AzureMachineLearningFileSystem],
                 transform: Optional[Union[TransformData, Callable]] = None,
                 default_transform_visualisation: Optional[bool] = False):
        """
        :param data_root: if provided as a string or Path, it is a local directory that contains folders with samples of data uniquely identifiable by their ID,
            otherwise it is an Azure Machine Learning filesystem instance referencinsg a specific location on Azure data storage.
        :param transform: Transformation pipeline of type TransformData or Albumentations callable to apply to images and masks.
        :param default_transform_visualisation: If True, a default transformation for visualisation purposes is applied to images and masks
            as specified in PH2Dataset.__init__.
        """
        self.data_root = data_root
        self.transform = transform
        self.default_transform_visualisation = default_transform_visualisation

        if self.transform is None and self.default_transform_visualisation is True:
            logging.getLogger(__name__).warning("No transform provided, applying default transform for visualisation.")
            self.transform = A.Compose([A.CenterCrop(height=510, width=510), A.ToTensorV2()])
            """Default transormation for visualisation purposes"""

        if isinstance(self.data_root, (str, Path)):
            self.azure_data = False
        elif isinstance(self.data_root, AzureMachineLearningFileSystem):
            self.azure_data = True
        else:
            raise ValueError("Provided data_root must be either a string or a Path pointing to a local directory with data \
            or an AzureMachineLearningFileSystem instance pointing to data in Azure")
                
        self.images_list, self.masks_list = get_ph2_data_paths(self.data_root, self.azure_data)
        """A tuple of two numpy arrays of dtype np.bytes_ that contains full paths to images and  masks"""


    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        """
        Return a dictionary containing an image and a mask. Augment if necessary, using the provided transform.

        :param item: The index or key used to access a specific sample of data from the dataset.
        :return sample: A dictionary with keys 'image' and 'mask' containing a tensor image and a tensor mask sample, respectively
            As per augmentations library internals, the output dimensions will be (C, H, W)  and (H, W) for images and masks, respectively,
        """
        if self.azure_data:
            # decode paths as required by Azure
            img_path = self.images_list[item]
            if isinstance(img_path, (bytes, np.bytes_)):
                img_path = img_path.decode("utf-8")
            
            mask_path = self.masks_list[item]
            if isinstance(mask_path, (bytes, np.bytes_)):
                mask_path = mask_path.decode("utf-8")

            # read data
            with self.data_root.open(img_path) as img:
                image = Image.open(img).copy()
            with self.data_root.open(mask_path) as msk:
                mask = Image.open(msk).copy()
        else:
            image = Image.open(self.images_list[item])
            mask = Image.open(self.masks_list[item])

        # Images and masks are of shape (H, W, 3) and (H, W), respectively,
        # where H and W vary slightly from sample to sample and on average H=576, W=768
       
        # (H, W, 3) and (H, W) shapes that are returned by PIL.Image are suitable for albumentations
    
        # transform data given an instance of TransformData
        if self.transform is not None and isinstance(self.transform, TransformData):
            # TransformData will ensure that the image and mask are in the correct format for albumentations 
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        # transform data given a manually specified albumentations transform or a default transform for visualisation purposes
        elif self.transform is not None and isinstance(self.transform, A.Compose):
            # albumentations require numpy arrays of shape (H,W,C), hence ensure_np_image
            # if torch.Tensor is used, it may require other shape
            image = ensure_np_image(image)
            mask = ensure_np_image(mask)
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        # if no transform is specified, convert to tensor using albumentations ToTensorV2
        # the latter requires image and mask to be numpy arrays 
        else:
            image_mask_tensor = A.ToTensorV2()(image=np.array(image), mask=np.array(mask))
            image = image_mask_tensor['image']
            mask = image_mask_tensor['mask']
        return {'image': image, 'mask': mask} # (3, H, W) and (H, W)
            
    def __len__(self):
        return len(self.images_list)

