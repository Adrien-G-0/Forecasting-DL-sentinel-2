# Forecasting corn NDVI through AI-based approaches using Sentinel 2 image time series

This repository allows you to train a model with different architectures and modalities to evaluate NDVI from satellite imagery.

## Architecture

### NewArchitectures
This class creates the model architecture with different data sources and aggregation types (early_fusion or middle_fusion).
All modalities are resized to be processed with a size of [channels, 256, 256].
- If fusion type is set to early_fusion, modalities are concatenated along the channel dimension
- Middle fusion is described in the middle_fusion_.py file

### Data Processing
- **Transformation**: min-max normalization
- **Train augmentation**: uniform rotation

### middle_fusion_.py
**Features**:
- Automatic creation of a CNN for each input source
- Transformation of inputs with dimension [channels, 256, 256] to [64, 256, 256]
- Concatenation of CNN outputs to form a fused tensor

### NewBase
Defines all functions for the Lightning module
Contains main function to launch the entire process

### NewDataset
- read_tif: allows reading .tif files and converting them to numpy format
- get_item: collects all modalities used according to conf['sources'] for an image

## Installation and Usage

Instructions to install dependencies and set up the project locally:

```bash
python -m venv .venv
# Then activate the environment
pip install -r requirements.txt
```

Create a params.json file based on the empty_params.json file at the root of the repository.

- `root_dir`: path to the folder containing train, val, and test folders
- `mean_dict_01`, `std_dict_01`, `max_dict_01`, `max_dict`, `min_dict`: paths used for min-normalization, required for all data types
- `method`: either "early_fusion" or "middle_fusion"
- `sources`: a list containing "rgb", "hs", "dem", "sar"

## Data Storage

Your data should be stored in a folder with the following structure:

```
Sources/
├── train/
│   └── image_1/
│       └── sar.tif
│       └── rgb.tif
│       └── dem.tif
│       └── hs.tif
│       └── ndvi.tif
├── val/
│   └── image_2/
│       └── sar.tif
│       └── rgb.tif
│       └── dem.tif
│       └── hs.tif
│       └── ndvi.tif
└── test/
    └── image_3/
        └── sar.tif
        └── rgb.tif
        └── dem.tif
        └── hs.tif
        └── ndvi.tif
```

Run the `NewArchitectures.py` file in the command terminal with the following arguments:

```bash
python NewArchitectures.py --json params.json --ckp checkpoint.ckp --test
```

The `--ckp checkpoint.ckp --test` parameters are optional.

The model is stored in the folder checkpoint/experiment_name/version