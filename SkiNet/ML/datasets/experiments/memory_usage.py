import uuid
from multiprocessing import shared_memory

import numpy as np
import psutil
import torch
from torch.utils.data import Dataset


def display_memory_consumption(data_loader, BATCH_SAMPLING_NUM):
    """
    Display memory consumption

    :param data_loader: DataLoader object
    :param BATCH_SAMPLING_NUM: Number of batches to sample to sample at for memory consumption, e.g. 10 for every 10th batch
    """
    mem_used=[]
    mem_used.append(psutil.virtual_memory().used/1024**3)
    
    for i, item in enumerate(data_loader):
        if i % BATCH_SAMPLING_NUM == 0:
            mem = psutil.virtual_memory()
            print(f'{i:8} - {mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')
            mem_used.append(mem.used/1024**3)
    return mem_used
    