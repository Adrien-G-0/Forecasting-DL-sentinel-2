import os
import torch
import pandas as pd
import numpy as np
import torch.utils.data as data
import rasterio as rs
import json

with open('path.json', 'r') as f:
    path_sources = json.load(f)


class NewDataset_ticino(data.Dataset):
    '''
    Permet de creer un dataset à partir des données stockées avec un dossier pour chaque modalité et un fichier csvfile qui gère les noms des images à selectionner
    '''
    

    def __init__(self, root_dir, csv_file, pca=False, trans=None):
        '''
        Dataset pour structure avec fichiers nommés par modalité
        Args:
            root_dir (str): Dossier racine contenant dossier de modalité.
            csv_file(csv): Odssier comprenant les indices des images à charger
            pca (bool): Pour future compatibilité PCA.
            trans (callable): Transformations à appliquer à (inputs, targets).
        '''
        # save
        self.fns = pd.read_csv(csv_file, names=['fns'], header=None)
        self.root_dir = root_dir
        self.pca = pca
        self.trans = trans

        # Définit les types de données attendus (ordre : inputs puis targets)
        
        self.input_types = ['RGB', 'pansh_data', 'DTM', 'SAR']
        self.target_types = ['ndvi'] 

    def __len__(self):
        return len(self.fns)

    # cv2 read unchanged
    def read_tif(self, sub_dir, fn):
        # define filename
        fn = os.path.join(self.root_dir, sub_dir, fn)
        # read tif
        data = rs.open(fn).read()
        # add channel
        if data.ndim < 3:
            data = np.expand_dims(data, axis=-1)
        return data

        # make them proper type
        num_modalities = 4
        imgs[:num_modalities] = [torch.from_numpy(cur_img).float() for cur_img in imgs[:num_modalities]]
        imgs[num_modalities:] = [torch.from_numpy(cur_img).long().squeeze(-1) for cur_img in imgs[num_modalities:]]

        if self.trans is not None:
            imgs = self.trans(imgs)

        inputs = imgs[:num_modalities]
        targets = imgs[num_modalities:]

        return inputs, targets, self.fns.iloc[idx]['fns']
    
        if self.trans is not None:
            inputs, targets = self.trans((inputs, targets))

        return inputs, targets, folder
    


class NewDatasetGlobal(data.Dataset):
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