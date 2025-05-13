import os
import torch
import pandas as pd
import numpy as np
import torch.utils.data as data
import rasterio as rs
import json

with open('path.json', 'r') as f:
    path_sources = json.load(f)
with open('params.json', 'r') as f:
    conf= json.load(f)
    
#Adaptation du nom des sources aux nom des fichier
# partie temporaire mmais simple pour alterner entre dtm, dem pour le type de terrain
# panshsdata pour les donnéees hyperspectrales
for i in range(len(conf["sources"])):
    if conf["sources"][i] == "hs" :
        conf["sources"][i]="pansh_data"
    if conf["sources"][i] == "dem" :
        conf["sources"][i]="DTM"
    if conf["sources"][i] == "sar" :
        conf["sources"][i]="SAR"
    if conf["sources"][i] == "rgb" :
        conf["sources"][i]="RGB"


class Dataset(data.Dataset):
    def __init__(self, root_dir, split='train', pca=False, trans=None):
        """
        Dataset pour structure avec fichiers avec une structure: dara/ train or test or val /image_ number / type of data.tif such as SAR...

        Args:
            root_dir (str): Dossier racine contenant les splits (train/val/test).
            split (str): Le sous-dossier à charger ('train', 'val', 'test').
            pca (bool): Pour future compatibilité PCA.
            trans (callable): Transformations à appliquer à (inputs, targets).
        """
        self.root_dir = os.path.join(root_dir, split)
        self.image_dirs = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])
        self.pca = pca
        self.trans = trans

        # Définit les types de données attendus (ordre : inputs puis targets)
        self.input_types = conf['sources']
        self.target_types = ['ndvi'] 

    def __len__(self):
        return len(self.image_dirs)

    def read_tif(self, filepath):
        with rs.open(filepath) as src:
            data = src.read()
        if data.ndim == 3:
            data = np.moveaxis(data, 0, -1)
        elif data.ndim == 2:
            data = np.expand_dims(data, axis=-1)
        return data

    def __getitem__(self, idx):
        folder = self.image_dirs[idx]
        folder_path = os.path.join(self.root_dir, folder)
        

        # loading input data
        inputs = []
        for t in self.input_types:
            path = os.path.join(folder_path, f"{t}.tif")
            
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Fichier d'entrée manquant : {path}")
            img = self.read_tif(path)
            inputs.append(torch.from_numpy(img).float())

        # loading target data
        targets = []
        for t in self.target_types:
            path = os.path.join(folder_path, f"{t}.tif")
            
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Fichier cible manquant : {path}")
            img = self.read_tif(path)
            
            targets.append(torch.from_numpy(img).float().squeeze(-1))
        
        if self.trans is not None:
            trans_img= self.trans([*inputs,*targets])
            
            inputs=trans_img[:len(inputs)]
            targets=trans_img[-len(targets)]
            
        return inputs, targets, folder
    

    
""" For the big dataset, onlmy the name input are different"""
'''
class NewDataset(data.Dataset):
    def __init__(self, root_dir, split='train', pca=False, trans=None):
        """
        Dataset pour structure avec fichiers nommés par image : XXXX_s2, XXXX_s1, XXXX_dsm, XXXX_worldcover,XXXX_ndvi

        Args:
            root_dir (str): Dossier racine contenant les splits (train/val/test).
            split (str): Le sous-dossier à charger ('train', 'val', 'test').
            pca (bool): Pour future compatibilité PCA.
            trans (callable): Transformations à appliquer à (inputs, targets).
        """
        self.root_dir = os.path.join(root_dir, split)
        self.image_dirs = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])
        self.pca = pca
        self.trans = trans

        # Définit les types de données attendus (ordre : inputs puis targets)

        self.input_types = ['s2', 's1', 'dsm','worldcover']
        self.target_types = ['ndvi'] 

    def __len__(self):
        return len(self.image_dirs)

    def read_tif(self, filepath):
        with rs.open(filepath) as src:
            data = src.read()
        if data.ndim == 3:
            data = np.moveaxis(data, 0, -1)
        elif data.ndim == 2:
            data = np.expand_dims(data, axis=-1)
        return data

    def __getitem__(self, idx):
        folder = self.image_dirs[idx]
        folder_path = os.path.join(self.root_dir, folder)
        base_name = folder.split('_')[-1]  # Exemple : '1963'

        # Lire les entrées
        inputs = []
        for t in self.input_types:
            path = os.path.join(folder_path, f"{base_name}_{t}.tif")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Fichier d'entrée manquant : {path}")
            img = self.read_tif(path)
            inputs.append(torch.from_numpy(img).float())

        # Lire les cibles
        targets = []
        for t in self.target_types:
            path = os.path.join(folder_path, f"{base_name}_{t}.tif")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Fichier cible manquant : {path}")
            img = self.read_tif(path)
            targets.append(torch.from_numpy(img).long().squeeze(-1))

        if self.trans is not None:
            inputs, targets = self.trans((inputs, targets))

        return inputs, targets, folder



'''