import os
import torch
import pandas as pd
import numpy as np
import torch.utils.data as data
import rasterio as rs
import json

# with open('params.json', 'r') as f:
#     conf= json.load(f)
    
#Adaptation du nom des sources aux nom des fichier

# for i in range(len(conf["sources"])):
#     if conf["sources"][i] == "dem" :
#         conf["sources"][i]="dtm"




class Dataset(data.Dataset):
    def __init__(self, root_dir,sources, split='train', pca=False, trans=None):
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
        self.input_types = list(sources).copy()
        self.target_types = ['ndvi'] 
        #Adaptation du nom des sources aux nom des fichier
        #TODO a améliore, soit change le nom des fichiers soit adapter le reste du code pour tout changer dem en dtm
        for i in range(len(self.input_types)):
            if self.input_types[i] == "dem" :
                self.input_types[i]="dtm"
        print('')
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
            if True and t=='lc':
                # Convertir l'image en one-hot
                img = self.onehot(img)
                img = img.astype(np.float32)
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
    

    def onehot(self, landcover_tif):
        """
        Convertit une image de classification en une représentation one-hot.
        Préserve le type d'entrée (numpy.ndarray ou torch.Tensor).
        
        Args:
            landcover_tif (numpy.ndarray ou torch.Tensor): Image de classification.
        
        Returns:
            Même type que l'entrée: Image one-hot encodée.
        """

    
        # Nombre de classes
        num_classes = 8  # 8 classes in the dataset
        
        if isinstance(landcover_tif, np.ndarray):
            # Cas NumPy
            one_hot = np.eye(num_classes)[landcover_tif]
            one_hot = one_hot.squeeze()
            return one_hot
        
        elif isinstance(landcover_tif, torch.Tensor):
        # Cas PyTorch
            
        
            # Supprimer la dernière dimension si elle est de taille 1
            if landcover_tif.shape[-1] == 1:
                landcover_squeezed = landcover_tif.squeeze(-1)
            else:
                landcover_squeezed = landcover_tif
            
            # Créer la matrice identité sur le bon device
            eye = torch.eye(num_classes, dtype=torch.float32)
            
            # One-hot encoding
            one_hot = eye[landcover_squeezed.long()]
            
            return one_hot
        



if __name__ == '__main__':
    ds= Dataset(root_dir='Sorted_Sources', sources=['rgb', 'hs', 'dem','sar','lc'], split='train',pca=False, trans=None)
    data= ds[0]
    print(data[0][4].shape)  # Affiche la forme de la première entrée
    lc= Dataset.onehot(ds,data[0][4])
    print(lc.shape)  # Affiche la forme de la première entrée