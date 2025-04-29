import os
import torch
import pandas as pd
import numpy as np
import torch.utils.data as data
import rasterio as rs
import json

# Charger le fichier JSON
with open('path.json', 'r') as f:
    path = json.load(f)


class Dataset(data.Dataset):

    def __init__(self, root_dir, csv_file, pca=False, trans=None):
        # save
        self.fns = pd.read_csv(csv_file, names=['fns'], header=None)
        self.root_dir = root_dir
        self.pca = pca
        self.trans = trans

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

    def __getitem__(self, idx):
        
        # Définition des sources
        sources = [
            path['source_rgb'], 
            path['source_hs'], 
            path['source_dem'], 
            path['source_sar'], 
            path['labeling_landuse'], 
            path['labeling_agriculture']
        ]  # TODO : Vérifiez si les chemins sont corrects
        
        # Chargement des données
        imgs = []
        for cur_source in sources:
            try:
                print(f"Chargement de la source : {cur_source}")
                
                imgs.append(self.read_tif(
                    cur_source, 
                    self.fns.iloc[idx]['fns']
                ))
                    
            except Exception as e:
                print(f"Erreur lors du chargement de {cur_source} : {e}")
                continue  # Permet de passer à la source suivante en cas d'erreur
        
        # Conversion au bon type
        
        num_modalities = 4
        try:
            imgs[:num_modalities] = [torch.from_numpy(cur_img).float() for cur_img in imgs[:num_modalities]]
            imgs[num_modalities:] = [torch.from_numpy(cur_img).long().squeeze(-1) for cur_img in imgs[num_modalities:]]
        except Exception as e:
            print(f"Erreur lors de la conversion des données : {e}")
            raise
        
        # Application des transformations si nécessaire
        if self.trans is not None:
            imgs = self.trans(imgs)
        
        # Séparation des entrées et cibles
        inputs = imgs[:num_modalities]
        targets = imgs[num_modalities:]
        
        return inputs, targets, self.fns.iloc[idx]['fns']














    # def __getitem__(self, idx):
    #     # defines sources
    #     sources = [
    #         path['source_rgb'], 
    #         path['source_hs'], 
    #         path['source_dem'], 
    #         path['source_sar'], 
    #         path['labeling_landuse'], 
    #         path['labeling_agriculture']
    #     ]
    #     # load them
    #     imgs = []
    #     for cur_source in sources:
    #         print(cur_source)
    #         if cur_source.split('/')[1] == 'HS':
    #             # imgs.append(np.load(os.path.join(self.root_dir, cur_source, self.fns.iloc[idx]['fns'].split('.')[0] + '.npy')))
    #             imgs.append(self.read_tif(path['source_pansh'], self.fns.iloc[idx]['fns'])) # read the pansh data TODO change
    #         else:
    #             imgs.append(self.read_tif(cur_source, self.fns.iloc[idx]['fns']))

    #     # make them proper type
    #     num_modalities = 3
    #     imgs[:num_modalities] = [torch.from_numpy(cur_img).float() for cur_img in imgs[:num_modalities]]
    #     imgs[num_modalities:] = [torch.from_numpy(cur_img).long().squeeze(-1) for cur_img in imgs[num_modalities:]]

    #     if self.trans is not None:
    #         imgs = self.trans(imgs)

    #     inputs = imgs[:num_modalities]
    #     targets = imgs[num_modalities:]

    #     return inputs, targets, self.fns.iloc[idx]['fns']
    