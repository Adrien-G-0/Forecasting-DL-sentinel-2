# Forecasting corn NDVI through AI-based approaches using Sentinel 2 image time series

This repository allows you to train a model with different architectures and modalities to evaluate NDVI from satellite imagery.

## Key Features

- **Multi-modal Fusion**: Combines data from various sources (RGB, hyperspectral, SAR, DTM, land cover, etc.)
- **Multiple Architectures**: Supports CNN-based and Transformer-based models
- **Flexible Fusion Strategies**: Early, middle, and late fusion approaches
- **Comprehensive Data Processing**: Includes normalization, augmentation, and encoding utilities


## Architecture

### TimeArchitecures
This improvement of the architecure of the model by adding the time as a modality, the time is implemented in the bottleneck to mak a linear transformation of the latent space before the decoder. Not implemented yet.

### TransformerArchitectures
A Vision Transformer encoder with CNN decoder, 
Swin Transformer zith CNN or UperNet decoder

### NewArchitectures
This class creates the model architecture with different data sources and aggregation types (early_fusion or middle_fusion).
All modalities are resized to be processed with a size of [channels, 256, 256].
- If fusion type is set to early_fusion, modalities are concatenated along the channel dimension
- Middle fusion is described in the middle_fusion_.py file
- Late Fusion is described in the late_fusion.py file

### Data Processing
- **Transformation**: min-max normalization
- **Train augmentation**: uniform rotation
- **Class-images**: OneHot encoding

## Fusion Strategies

### Middle Fusion (middle_fusion.py)
**Features**:
- Automatic creation of a CNN for each input source
- Transformation of inputs with dimension [channels, 256, 256] to [to_determine, 256, 256]
    ```markdown
    The number of channels for each step of the transformation can be specified in the `params.json` file under the `conf_'source'` key.
    ```
- Concatenates processed features into a fused tensor

### Late Fusion (late_fusion.py).py
**Features**:
- Automatic creation of a CNN for each input source
- Create an encoding CNN for each input source, from size [source_channels, 256, 256] to [source_channels* 2⁴, 16, 16]
- Concatenate the features in source_channels
- Recreate the targeted imaged trough simple CNN decodeur

### NewBase
Defines all functions for the Lightning module
Contains main function to launch the entire process

### NewDataset
- read_tif: allows reading .tif files and converting them to numpy format
- get_item: collects all modalities used according to conf['sources'] for an image

## Data Processing Pipeline

Comprehensive data handling utilities in NewDataset:
- **Image Reading**: `read_tif` converts .tif files to numpy format
- **Normalization**: Min-max normalization for all modalities
- **Augmentation**: Uniform rotation for training data
- **Encoding**: OneHot encoding for class images
- **Multi-modal Loading**: `get_item` collects specified modalities from params.json

## Installation and Usage

Instructions to install dependencies and set up the project locally:

```bash
python -m venv .venv
# Then activate the environment
pip install -r requirements.txt
```

Create a params.json file based on the empty_params.json file at the root of the repository for a CNN model; for a SwinTransformer, use the empty_Tparams.json

- `root_dir`: path to the folder containing train, val, and test folders
- `mean_dict_01`, `std_dict_01`, `max_dict_01`, `max_dict`, `min_dict`: paths used for min-normalization, required for all data types
- `method`: either "early_fusion" , "middle_fusion" , "late_fusion"
- `sources`: a list containing "rgb", "hs", "dtm", "sar", "lc", "sau","esa"


## Data Storage

Your data should be stored in a folder with the following structure:
all files names are : source.tif 
```
Sources/
├── train/
│   └── image_1/
│       └── lc.tif
│       └── sau.tif
│       └── sar.tif
│       └── rgb.tif
│       └── dtm.tif
│       └── hs.tif
│       └── esa.tif
│       └── ndvi.tif
│       └── ...
├── val/
│   └── image_2/
│       └── sar.tif
│       └── rgb.tif
│       └── dtm.tif
│       └── hs.tif
│       └── ndvi.tif
│       └── ...
└── test/
    └── image_3/
        └── sar.tif
        └── rgb.tif
        └── dtm.tif
        └── hs.tif
        └── ndvi.tif
│       └── ...
```

A main static method is available the `NewArchitectures.py` file tha can run with  the command terminal with the following arguments:



```bash
python NewArchitectures.py --json params.json --ckp checkpoint.ckp --test
```

The `--ckp checkpoint.ckp --test` parameters are optional.

Model are stored in the folder checkpoint/experiment_name/version