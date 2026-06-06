# Data

## ISIC 2017 dataset

ISIC 2017 is a dermoscopic segmentation dataset with predefined train/val/test splits. This is the primary dataset used in this project to train, validate and test the presented UNet2D segmentation model. The project respects these official splits and does not re-shuffle them.

- Copyright:
Codella N, Gutman D, Celebi ME, Helba B, Marchetti MA, Dusza S, Kalloo A, Liopyris K, Mishra N, Kittler H, Halpern A. "Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC)". [arXiv: 1710.05006](
https://doi.org/10.48550/arXiv.1710.05006)


- Dataset can be downloaded at [ISIC website](https://challenge.isic-archive.com/data/#2017): https://challenge.isic-archive.com/data/#2017
- Alternatively, there are pre-downloaded datasets available for use in e.g. Kaggle



### Download example for ISIC-2017 residing on Kaggle for dev work on Lightning Studio

```bash
#install kaggle cli tools if has not done so yet
pip install --quiet kaggle
```

```bash
#!/bin/bash
# download the dataset into out_dir
OUT_DIR="/teamspace/studios/this_studio/isic2017"
kaggle datasets download -d johnchfr/isic-2017 -p $OUT_DIR --unzip
```

### Generate metadata CSV

```bash
python -m SkiNet.ML.datasets.preprocessing.metadata_csv_factory \
  --dataset-key-str ISIC2017 \
  --local-data-root /teamspace/studios/this_studio/isic2017
```


### Config

```yaml
GENERAL_CONFIG:
  dataset: "isic2017_dataset"
DATA_CONFIG:
  local_data_root: "/teamspace/studios/this_studio/isic2017"
  azure_data: False
```

> **Note:** ISIC 2017 is large enough that `cache_in_ram: true` may exhaust available RAM on a lightning machine. Set `cache_in_ram: false` in `TRAIN_CONFIG` for this dataset.


## PH2 Dataset

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

### Data format

- Original images are .bmp and were converted to .png to be able to read them with torchvision

### Make a new metadata file

- Original data contain a text file, PH2_dataset.txt, listing e.g. sample ID, clinical diagnosis, colour of lesions, etc.
- Path to data are generated from the known file structure using ```{py:class} SkiNet.ML.datasets.preprocessing.PH2MetadataFactory```
- The search pattern is specified in ```{py:class} SkiNet.Utils.project_paths```
- Metadata can be created via

Example for local data:
```
python metadata_csv_factory.py --dataset_key_str="PH2" --local_data_root="path/to/local/data/PH2folder"
```

where PH2folder is the root_dir as per file structure above and the command will generate a metadata CSV "ph2_metadata.csv" in the root folder.

Example for Azure data:
```
python metadata_csv_factory.py --dataset_key_str="PH2" --azure_data,
```
where no path for the root_dir is required and the command will generate a metadata CSV "ph2_metadata.csv" in the root folder on Azure for that dataset (as specified in ```azure_settings.yaml```)

### Accessing the Metadata DataFrame

`BaseDataConfig` exposes dataset metadata as a pandas DataFrame through the `metadata` property.
The mechanism is **lazy** (no I/O at construction time), **validated** (required columns are checked on
first load), and **environment-aware** (transparently handles local paths and Azure Blob Storage).

#### Lazy loading

The first call to `.metadata` triggers a CSV read and column validation.
Subsequent calls return the in-memory cache immediately:

```python
cfg = PH2DatasetConfig(local_data_root="/data/PH2")

df = cfg.metadata   # reads CSV from disk, validates columns, caches result
df = cfg.metadata   # returns cached DataFrame — no disk I/O
```

The underlying cache is stored in the private attribute `_metadata` (`PrivateAttr`, default `None`).
Because it is a `PrivateAttr` it is excluded from Pydantic validation and serialization; only the
config fields (paths, split sizes, etc.) are persisted or validated.

#### Deepcopy behaviour

`BaseDataConfig` overrides `__deepcopy__` to **reset `_metadata` to `None`** in every copy.
The regular Pydantic model fields (paths, flags, split parameters) are deep-copied as normal;
only the transient cache is cleared.

This matters during hyperparameter sweeps where `deepcopy(main_config)` is called once per trial.
Without the override, a DataFrame loaded before the sweep would be duplicated in full into every
trial's config — the same data in memory N times. With the override each trial's copy starts with
`_metadata = None` and lazy-loads independently on first access:

```python
from copy import deepcopy

# metadata loaded once on the template config (e.g. during a pre-flight check)
_ = main_config.dataconfig.metadata

# each trial gets a fresh copy with _metadata = None
trial_cfg = deepcopy(main_config)
assert trial_cfg.dataconfig._metadata is None   # no DataFrame duplication

# lazy-load happens on first access within the trial, as normal
df = trial_cfg.dataconfig.metadata
```

Note: `ClassVar` attributes (`METADATA_CSV_NAME`, `REQUIRED_COLUMNS`, `DATASET_KEY`) live on the
**class**, not the instance, so deepcopy never touches them — they are shared constants and
behave correctly without any special handling.

#### Required subclass configuration

Subclasses must declare three class-level attributes for metadata loading to work:

```python
class MyDatasetConfig(BaseDataConfig):
    METADATA_CSV_NAME: ClassVar[str] = "metadata.csv"       # filename of the CSV in data_root
    REQUIRED_COLUMNS: ClassVar[Set[str]] = {"id", "label"}  # columns that must be present
    DATASET_KEY: ClassVar[Optional[DatasetKey]] = DatasetKey.MY_DATASET  # used for Azure path resolution
```

A `ValueError` is raised on first `.metadata` access if any required column is absent or all
required columns contain only empty values.
