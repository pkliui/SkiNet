# Note about modification to PyTorch's default dataloader

This section describes modifications to PyTorch's default DataLoader used in SkiNet to prevent
spawning new processes at the beginning of each epoch.

A Jupyter notebook with examples is available at:
`SkiNet/ML/dataloaders/examples/RepeatDataloaders.ipynb`

---

## Motivation

Spawning new worker processes at the beginning of each epoch in PyTorch's DataLoader
(when `num_workers > 0` and `persistent_workers=False`) causes:

- Overhead of initialising the dataset in each worker (by deserialising from the main process)
- Memory overhead as each worker requires its own memory space
- Any worker-cached state is lost when workers are shut down
- Copy-on-write behaviour during fork increases memory if the dataset contains Python lists or dicts
- Potential deadlocks and race conditions with shared resources

---

## Dataloaders and subprocesses

When iterating over a DataLoader:

1. **Dataset initialisation** runs in the main process (`Dataset.__init__`).
2. **DataLoader initialisation** runs in the main process (`DataLoader.__init__`).
3. **Prefetching & Queues:** At the start of epoch 0, PyTorch spawns `num_workers` separate
   subprocesses. Dataset indices are put in worker input queues (up to `prefetch_factor × num_workers`
   indices ahead). Each worker calls `__getitem__` asynchronously; the main process collects results.

At the start of each epoch, `for batch in loader` calls `iter(loader)`, which calls `__iter__()`:

- If `persistent_workers=True` and `num_workers > 0` and no iterator exists yet, `_get_iterator()`
  creates a new iterator.
- If an iterator exists (epoch > 0), its state is reset via `_reset()` — workers are reused.
- If `persistent_workers=False` and `num_workers > 0`, `_get_iterator()` is called every epoch,
  creating **new workers each time**.

When `num_workers > 0`, `_get_iterator()` returns `_MultiProcessingDataLoaderIter(self)`. New workers
are created in `_MultiProcessingDataLoaderIter.__init__()`. Workers are normally shut down when the
sampler is exhausted at epoch end.

---

## How SkiNet prevents new worker spawning: RepeatDataLoader

`SkiNet.ML.dataloaders.dataloaders.RepeatDataLoader` subclasses `torch.utils.data.DataLoader` and
relies on a single mechanism — **persistent workers**. This replaces an earlier
`_RepeatSampler`/iterator-override implementation (the previous infinite-sampler approach is gone).

In `RepeatDataLoader.__init__`:

- `persistent_workers` defaults to `num_workers > 0` (set only if the caller did not pass it). With
  `num_workers=0` the flag is invalid and PyTorch raises, so it is left `False`. Persistent workers
  survive across epochs: PyTorch's `_MultiProcessingDataLoaderIter` is created once and `_reset()`
  between epochs instead of being torn down and respawned.
- `worker_init_fn` defaults to `default_worker_init_fn`, which sets `cv2.setNumThreads(0)` and seeds
  `numpy`/`random` per worker from `torch.initial_seed() % 2**32` for reproducibility.
- `collate_fn` defaults to `collate_preserving_specs` (see below).
- The dataset must be sized (`__len__`), else `TypeError` is raised.
- `max_num_to_repeat` is accepted for backward compatibility but **ignored** — Lightning's `Trainer`
  controls epoch iteration via `max_epochs`.

---

## Usage with ISIC 2017

`create_segmentation_dataloaders(main_config)` is the segmentation entry point: it calls
`create_segmentation_datasets_from_config` then `create_dataloaders_from_datasets`, which builds the
`RepeatDataLoader` instances for the train, val, and test splits:

```python
from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml
from SkiNet.ML.dataloaders.create_dataloaders import create_dataloaders_from_datasets
from SkiNet.ML.datasets.dataset_factory import create_segmentation_datasets_from_config

cfg = load_config_from_yaml("main_config.yaml")
datasets = create_segmentation_datasets_from_config(cfg)
loaders = create_dataloaders_from_datasets(datasets, cfg.trainconfig)

# loaders.train, loaders.val, loaders.test are RepeatDataLoader instances
```

Key dataloader settings from `TrainConfig` that affect worker behaviour:

| Field | Effect |
|---|---|
| `num_workers` | Auto-set to `os.cpu_count()` for single-GPU; divided by device count for DDP |
| `pin_memory` | Auto-set `True` on CUDA, `False` on MPS/CPU |
| `prefetch_factor` | Batches pre-loaded per worker; `None` when `num_workers=0` |
| `cache_in_ram` | Eliminates disk I/O in workers when `True` (recommended for small datasets) |

---

## Custom collate function

`RepeatDataLoader` uses `collate_preserving_specs` as its `collate_fn`. This preserves the
`specs` field (sample metadata) as a Python list rather than attempting to stack it into a tensor,
since metadata values are heterogeneous strings.
