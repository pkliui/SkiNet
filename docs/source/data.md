# Data

## PH2 Dataset

### Data format

- Original images are .bmp and were converted to .png to be able to read them with torchvision

### Metadata

- Original data contain a text file, PH2_dataset.txt, listing e.g. sample ID, clinical diagnosis, colour of lesions, etc.
- Path to data are generated from the known file structure using ```{py:class} SkiNet.ML.datasets.preprocessing.PH2MetadataFactory```
- The search pattern is specified in ```{py:class} SkiNet.Utils.project_paths```
- Metadata can be created via

Example for local data:
```
python metadata_csv_factory.py --dataset_key_str="PH2" --local_data_root="path/to/local/data/PH2folder"
```

Example for Azure data:
```
python metadata_csv_factory.py --dataset_key_str="PH2" --azure_data,
```

