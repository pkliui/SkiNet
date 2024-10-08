
import numpy as np
from pathlib import Path
from torch import randint
from torch.utils.data.dataset import Dataset



def get_random_sample(data_set: Dataset) -> dict[np.array, np.array]:
    """
    Function to return a random pair of an image and a mask having the same sample_idx

    :return dictionary containing an image array and a mask array randomly selected from the provided dataset

    """
    sample_idx = randint(len(data_set), size=(1,)).item()

    img = data_set[sample_idx]['image']
    mask = data_set[sample_idx]['mask']
    sample_name = Path(data_set.images_list[sample_idx]).parent.parent.name

    return {'image': img, 'mask': mask, 'name': sample_name}