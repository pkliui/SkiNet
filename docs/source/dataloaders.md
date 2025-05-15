# Dataloaders

- This document describes modifications to PyTorch's default DataLoader used in SkiNet to prevent spawning new processes at the beginning of each epoch.

- A Jupyter notebook with examples is available here: [RepeatDataloaders Example Notebook](../SkiNet/Sandbox/RepeatDataloaders.ipynb)

  ### Dataloaders and subprocesses 

- Let us have a look what happens when we start iterating over a dataloader:
  - **Dataset initialisation** begins with the ```Dataset.__init__()``` method being called in the main process
  - **Dataloader initialisation** is done in the ```Dataloader.__init__()```  method

  - **Prefetching & Queues:** Pytorch uses multiprocessing library and exactly at the beginning of epoch 0, when we start iterating over epochs, *Pytorch spawns ```num_workers > 1 ``` separate subprocesses (with their own PIDs) to handle loading the data*. 
  
  - At this point, the dataset indices are  sent to the workers. This “prefetching” is done by putting up to (prefetch_factor × num_workers) indices in the worker input queues. This happens before the first batch is actually yielded (see *Dataloader with persistent workers*  and *Dataloader with RepeatSampler* sections in [RepeatDataloaders Example Notebook](../SkiNet/Sandbox/RepeatDataloaders.ipynb))

  - Each worker process picks up an index from its queue and calls getitem to fetch the data and does this asynchronously. The main process collects these results from a shared result queue and the individual samples are eventually being grouped into batches by the DataLoader. E.g. the pre-collected indicies from epoch 1 may be seen in batches yielded in epoch 2

   
  - **What happens under the hood at the beginning of each epoch:** At the beginning of each epoch, we call ```for batch in loader``` and under the hood Pytorch calls

    ```
    iterator = iter(loader)
    batch = next(iterator
    ```

  - At this point, the ```__iter__()``` method  distinguishes between two scenarios based on whether persistent workers are enabled and if there are any worker processes (i.e., ```num_workers > 0```). 
    - If ```persistent_workers=True``` and ```num_workers > 0``` and the iterator does not already exist (e.g. ```self._iterator``` is None at epoch 0), the method  ```_get_iterator()``` is called to create a new iterator. 
    - However, if an iterator exists, it is not recreated; instead, its state is reset by invoking its ```_reset``` method with the current loader instance. In essense, in this case, the iterator is created only once during the lifetime of the DataLoader so that the worker processes can be reused across iterations. 
    - If ```persistent_workers=False``` and ```num_workers > 0```,  the ```_get_iterator()``` method is called to create a new iterator (and hence new workers!).

  - If ```num_workers > 0```,  ```_MultiProcessingDataLoaderIter(self)``` is returned:

    ```
    def _get_iterator(self) -> "_BaseDataLoaderIter":
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)
    ```
  
    and **new workers are created in the``` _MultiProcessingDataLoaderIter.__init__()``` method at the beginning of each epoch** (unless we prevent this from happening by some means).

  - Each worker obtains its own Dataset instance (deserialized from the main one via pickle)

  - And **each dataloader receives a sampler** that is used to determine which data indices to extract from the dataset and in what order.  *Normally, the workers are shut down when the sampler is exhausted in the``` _MultiProcessingDataLoaderIter._next_data()``` method* (unless we prevent this from happening by some means).
  
  
  ### How to prevent Pytroch from spawning new processes for each new epoch? 

- To prevent spawning new processes, we should ensure two things:
  1. **Re-use the iterator in our Dataloader across all epochs**, i.e. whenever a new epoch is started and __iter__ is called, there should be the same iterator available. 
  2. **Provide an infinite sampler** that would prevent us from shutting down the workers once the sample is exhausted, i.e. at the end of an epoch
  
  #### Re-use of the iterator
- In SkiNet, this is done by specifying ```self.iterator = None``` in the ```_RepeatDataloader.__init__()``` method and then calling the ```Dataloader.__iter__()``` only ONCE at epoch 0  in ```_RepeatDataloader.__iter__()```  (as at all other epochs it will not be None):
  
    ```
    def __iter__(self) -> Any:
        if self.iterator is None:
            self.iterator = super().__iter__()  # type: ignore
    ```


  #### Inifinite sampler

- Normally, if no persistent workers are set and the iterator's sampler is exhausted, no new index can be delivered to the iterator. In this case,```StopIteration``` is raised and the code shuts down the workers. See  ```_MultiProcessingDataLoaderIter._next_data()```.
  
- To make sure samples are yielded indefinitely by the iterator, I extend Pytorch's BatchSampler so that it never runs out of indicies being supplied to the iterator. This is done in the ```SkiNet.ML.dataloaders.dataloaders._RepeatSampler.__iter__()```  class: 
- 
```
def __iter__(self) -> Any:
    num_to_repeat = 0
    # __iter__ is called each time we start iterating over the data loader, such as at the beginning of each epoch
    # i.e. calling iter on the data loader leads to calling __iter__ on the sampler, see _BaseDataLoaderIter.__init__()
    while self.max_num_to_repeat == 0 or num_to_repeat < self.max_num_to_repeat:
        print("Iter of sampler in RepeatSampler.__iter__   ", iter(self.sampler))
        yield from iter(self.sampler)
        # this is being incremented each epoch
        num_to_repeat += 1
```

- The ```_RepeatSampler.__iter__()``` method implements the repeating logic, where it keeps yielding indicies from the sampler either indefinitely or a specified max number of times (as required by the number of epochs). When one calls ```__iter()__``` on a DataLoader, it eventually constructs a DataLoader iterator (for example, a ```_SingleProcessDataLoaderIter``` or ```_MultiProcessingDataLoaderIter```). In the initialization of that iterator, in the ```_BaseDataLoaderIter.__init__``` method, it calls

```
  self._sampler_iter = iter(self._index_sampler)
```

- If auto-collation is enabled (which is the case when a batch_sampler is used), ```self._index_sampler``` simply returns the ```batch_sampler ```that was passed to the DataLoader. In other words, calling ```iter(loader)``` leads to calling ```iter()``` on the ```batch_sampler```. And that’s where the iter method of  _RepeatSampler is invoked providing indicies to the iterator.