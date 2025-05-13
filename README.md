# Forecasting corn NDVI through AI-based approaches using sentinel 2 image time series
Ce repo permet d'entrainer un modèle avec différentes architectures et modalités possibles pour évaluer le NDVI d'image satellite

#### NewArchitectures 
    Crée l'architecture du modèle avec les différentes sources, et le type d'aggrégation de donnée, early ou middle.
    toutes les modalités sont redimensionnées pour être traitées avec une taille de [channels,256,256]
    si le type de fusion est mis sur early_fusion, les modalité sont concaténé sur la dimension de channels
        middle fusion, décris dans descitpion fichier middle_fusion

# transformation
    min-max
# augmentation pour le train
    rotation uniforme 

#### middele_fusion_.py
    ## Fonctionnalités
    - Création automatique d'un CNN pour chaque source d'entrée.
    - Transformation des entrées de dimension `[channels, 256, 256]` en `[64, 256, 256]`.
    - Concatenation des sorties CNN pour former un tenseur fusionné.

#### NewBase
définie toutes les fonctions pour le module Lignhntning module
fonction mail pour lancer le tout
#### Newdataset
    read_tif pour permettre de lire les fichiers .tif pour les mettre en format numpy
    get_item permet de recolter toutes les modalités utilisées selon conf['sources']
## Installation et utilisation

Instructions pour installer les dépendances et configurer le projet localement:
```bash

python - m venv .venv

Puis activer l'environnement

pip install -r requirements.txt
```
créer un fichier params.json à partir du fichier empty_params.json à mettre à la base du répertoire.
```root_dir``` correspond au chemin d'accès au dossier qui contient les dossier train val et test
```mean_dict_01,std_dict_01,max_dict_01,max_dict,min_dict```: are path to use the min-normalisation and and needed for all the data types
```method```: is either "early_fusion" of "middle_fusion"
```sources```: is a list containing "rgb","hs,"dem,"sar"

Stockage des données:
Vos données doivent être stcokées dans un dossier sous la forme:
```bash
Sources/
├── train/
│   └── image_1/
│       └── sar.tif
        └── rgb.tif
        └── dem.tif
        └── hs.tif
        └── ndvi.tif
├── val/
│   └── image_2/
│       └── sar.tif
        └── rgb.tif
        └── dem.tif
        └── hs.tif
        └── ndvi.tif
└── test/
    └── image_3/
        └── sar.tif
        └── rgb.tif
        └── dem.tif
        └── hs.tif
        └── ndvi.tif
```


Lancer le fichier `NewArchitectures.py` dans le terminal de commande avec les arguments suivants :
```bash
python NewArchitectures.py --json params.json --ckp checkpoint.ckp --test
```
```--ckp checkpoint.ckp --test``` sont factultatifs


Le modèle est stocké dans le dossier checkpoint/experiment_name/version