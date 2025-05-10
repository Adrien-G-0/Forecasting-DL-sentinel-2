# Nom du Projet
Forecasting corn NDVI through AI-based approaches using sentinel 2 image time series


## Table des Matières
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Lancement du Programme] (#lancement-du-programme)
- [Contributions](#contributions)
- [Tests](#tests)
- [Licence](#licence)

## Utilisation
mettre toutes les relative path dans path.json au bon endroit en prenant empty_path comme exemple
#### NewArchitectures 
    Crée l'architecture du modèle avec les différentes sources, et le type d'aggrégation de donnée, early ou middle

#### middele_fusion_.py
Lorsque que le mode de fusion middle fusion est sélectionner, en fonction des sources utilisées dans le modèle, il est créé un petit CNN qui transfort chaque entrée de taille [channels,256,256] en [64,256,256]. Les sorties des CNN sont concaténées pour former un tenseur de la forme [64* nbre de source,256,256]

#### NewBase
#### Newdataset
## Installation

Instructions pour installer les dépendances et configurer le projet localement:
```bash

python - m venv .venv

Puis activer l'environnement

pip install -r requirements.txt

creer un fichier params à partir du fichier empty_params.json à mettre à la base du répertoire.

Stockage des données:
Vos données doivent être stcokées dans un dossier sous la forme:
Sources/train/image_nbre/data_source.tif
