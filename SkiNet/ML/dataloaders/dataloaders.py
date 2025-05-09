import logging
from math import ceil
from typing import Any

from torch.utils.data import (BatchSampler, DataLoader, Dataset, RandomSampler,
                              Sampler, SequentialSampler)


class _RepeatSampler(BatchSampler):

    """
    A sampler that repeats the Pytorch's BatchSampler indefinitely or a given number of times. 

    In combination with a modified Pytorch's Dataloader, this sampler does not result in starting new worker processes at the beggining of a new epoch.
    This is because the workers are shut down when the sampler is exhausted: see the _MultiProcessingDataLoaderIter._next_data method.

    Adapted from https://github.com/pytorch/pytorch/issues/15849

    If `max_num_to_repeat` is set to 0 (default), the sampler will repeat indefinitely. 
    Otherwise, it will repeat a finite number of times as specified by `max_num_to_repeat`.

    :param sampler: The sampler to repeat, e.g. BatchSampler
    :param batch_size: The number of samples per batch.
    :param drop_last: If True, drop the last incomplete batch. Default is False.
    :param max_num_to_repeat: The maximum number of times to repeat the sampler. Default is 0 (infinite).
        Typically max_num_to_repeat should be equal to the number of epochs.
    """

    def __init__(self, 
                 sampler: Sampler, 
                 batch_size: int, 
                 drop_last: bool = False, 
                 max_num_to_repeat: int = 0) -> None:

        super().__init__(sampler, batch_size, drop_last)
        self.sampler = sampler
        self.max_num_to_repeat = max_num_to_repeat

    def __iter__(self) -> Any:
        num_to_repeat = 0
        # __iter__ is called each time we start iterating over the data loader, such as at the beginning of each epoch
        # i.e. calling iter on the data loader leads to calling __iter__ on the sampler, see _BaseDataLoaderIter.__init__
        while self.max_num_to_repeat == 0 or num_to_repeat < self.max_num_to_repeat:
            logging.getLogger(__name__).debug(f"Iter of sampler in RepeatSampler.__iter__   {iter(self.sampler)}")
            # this is being incremented each epoch
            num_to_repeat += 1
            yield from iter(self.sampler)



class RepeatDataLoader(DataLoader):
    """
    This class implements a data loader that avoids spawning a new process after each epoch.
    It is a subclass of PyTorch's DataLoader and uses a custom BatchSampler to repeat the dataset indefinitely or a given number of times.
    
    Adapted from https://github.com/pytorch/pytorch/issues/15849
    """

    def __init__(self,
                 dataset: Dataset,
                 max_num_to_repeat: int = 10,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 drop_last: bool = False,
                 **kwargs: Any):
        """
        :param dataset: The dataset that should be loaded.
        :param max_num_to_repeat: The maximum number of times the dataset should be repeated. If set to 0, the dataset will be repeated indefinitely.
        :param batch_size: The number of samples per minibatch. Default is 1.
        :param shuffle: If True, the dataset will be shuffled randomly using RandomSampler, otherwise SequentialSampler will be used. Default is False.
        :param drop_last: If True, drops incomplete minibatches at the end.
        :param kwargs: Any additional arguments that will be passed through to the Dataloader constructor.
        """
        self.dataset = dataset
        self.max_num_to_repeat = max_num_to_repeat
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # specify the sampler
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

        # create a batch sampler to yield mini-batches of indicies
        self.repeatdl_batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        logging.getLogger(__name__).debug(f"BatchSampler repeatdl_batch_sampler length {len(self.repeatdl_batch_sampler)}")     
        logging.getLogger(__name__).debug(f"BatchSampler repeatdl_batch_sampler {self.repeatdl_batch_sampler}")     

        # create a repeat sampler that will repeat batch sampler forever or a given number of times
        repeat_sampler = _RepeatSampler(self.repeatdl_batch_sampler, batch_size=batch_size, max_num_to_repeat=max_num_to_repeat)
        logging.getLogger(__name__).debug(f"BatchSampler repeat_sampler length {len(repeat_sampler)}")     
        logging.getLogger(__name__).debug(f"BatchSampler repeat_sampler {repeat_sampler}")   

        # create the data loader with the repeat sampler
        # NB! This sets self.batch_sampler to the repeat sampler! Caution whilst specifying the length of RepeatDataLoader in __len__ !
        super().__init__(dataset=dataset, batch_sampler=repeat_sampler, **kwargs)
        
        # do not initialise the iterator here, but do this in __iter__
        self.iterator = None
        logging.getLogger(__name__).debug(f"Iterator in RepeatDataloder.__init__ {self.iterator}")   

    def __len__(self) -> int:
        # NB! We use self.repeatdl_batch_sampler here  as it reflects the actual length of the dataset,
        # in contrast to the self.batch_sampler provided by infinitely sampling super().__init__
        return len(self.repeatdl_batch_sampler)

    def __iter__(self) -> Any:
        # Normally PyTorch creates worker processes every time once we enter Dataloader.__iter()__
        # When __iter__ is called for the first time, self.iterator is None and we create a new iterator. 
        # This is the first time when we enter the worker process.
        # With the logic below, in all subsequent calls to __iter__, self.iterator is not None and we are just re-using it.
        # This way we avoid spawning a new process every time we enter __iter__.

        if self.iterator is None:
            logging.getLogger(__name__).debug(f"Iterator in RepeatDataloder.__iter__ is None   {self.iterator}")   
            # create an iterator here if it is still None and re-use it for the next __iter__ call
            # essentially, this line will be called only ONCE whilst prefetching data (even before epoch 0)
            # for all subsequent epochs, the iterator will not be None and will be re-used
            self.iterator = super().__iter__()  # type: ignore
            logging.getLogger(__name__).debug(f" Iterator in RepeatDataloder.__iter__     {self.iterator}")   
        else:
            logging.getLogger(__name__).debug(f" Iterator in RepeatDataloder.__iter__  is being reused    {self.iterator}")   
        for i in range(len(self)):
            yield next(self.iterator)
