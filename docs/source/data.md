# Data

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

### Metadata

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
