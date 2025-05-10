# Nom du Projet
Forecasting corn NDVI through AI-based approaches using sentinel 2 image time series


## Table des Matières
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Lancement du Programme] (#lancement-du-programme)
- [Contributions]
- [Licence](#licence)

## Utilisation
mettre toutes les relative path dans path.json au bon endroit en prenant empty_path comme exemple
#### NewArchitectures 
    Crée l'architecture du modèle avec les différentes sources, et le type d'aggrégation de donnée, early ou middle.
    toutes les modalité sont redimensionnées pour être traitées avec une taille de [channels,256,256]
    si le type de fusion est mis sur early_fusion, les modalité sont concaténé sur la dimension de channels
        middle fusion, décris dans descitpion fichier middle_fusion

# transformation
    min-max
# augmentation pour le train
    rotation

#### middele_fusion_.py
    ## Fonctionnalités
    - Création automatique d'un CNN pour chaque source d'entrée.
    - Transformation des entrées de dimension `[channels, 256, 256]` en `[64, 256, 256]`.
    - Concatenation des sorties CNN pour former un tenseur fusionné.

#### NewBase
définie toutes les fonctions pour le module LIgnhntning module
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
cééer un fichier params.json à partir du fichier empty_params.json à mettre à la base du répertoire.

Stockage des données:
Vos données doivent être stcokées dans un dossier sous la forme:
```bash
Sources/
├── train/
│   └── image_1/
│       └── sar.tif
        └── dem.tif
        └── hs.tif
        └── ndvi.tif
├── val/
│   └── image_2/
│       └── sar.tif
        └── dem.tif
        └── hs.tif
        └── ndvi.tif
└── test/
    └── image_3/
        └── sar.tif
        └── dem.tif
        └── hs.tif
        └── ndvi.tif
```


Lancer le fichier NewArchitecture
