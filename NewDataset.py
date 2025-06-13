import os
import torch
import pandas as pd
import numpy as np
import torch.utils.data as data
import rasterio as rs
import json

with open('params.json', 'r') as f:
     conf= json.load(f)
# Used only to define the number of classes for one-hot encoding




class Dataset(data.Dataset):
    def __init__(self, root_dir,sources, split='train', pca=False, trans=None):
        """
        Dataset for structure with files with structure: dara/ train or test or val /image_ number / type of data.tif such as sar...

        Args:
        root_dir (str): Root folder containing splits (train/val/test).
        split (str): The sub-folder to load ('train', 'val', 'test').
        pca (bool): For future PCA compatibility.
        trans (callable): Transformations to apply to (inputs, targets).
        """

        self.root_dir = os.path.join(root_dir, split)
        self.image_dirs = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])
        self.pca = pca
        self.trans = trans

        # Defines the expected data types (order: inputs then targets)
        self.input_types = list(sources).copy()
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
                raise FileNotFoundError(f"Missing file : {path}")
            img = self.read_tif(path)
            if t=='lc':
                # Onehot encoding 
                img = self.onehot(img,conf['num_class_lc'])
                img = img.astype(np.float32)
            if t=='sau':
                # Onehot encoding 
                img = self.onehot(img,conf['num_class_sau'])
                img = img.astype(np.float32)
            if t=='esa':
                # Onehot encoding
                if isinstance(img, np.ndarray):
                    img = img // 10
                elif isinstance(img, torch.Tensor):
                    img = img // torch.tensor(10, dtype=img.dtype)
                img = self.onehot(img,conf['num_class_esa'])
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
    

    def onehot(self, cover_tif,num_class):
        """
        Converts a classification image into a one-hot representation.
        Preserves the input type (numpy.ndarray or torch.Tensor).
        
        Args:
        cover_tif (numpy.ndarray or torch.Tensor): Classification image.
        
        Returns:
        Same type as input: Encoded one-hot image.
        """  
        if isinstance(cover_tif, np.ndarray):
            # NumPy
            cover_tif= np.array(cover_tif, dtype=np.uint8)
            one_hot = np.eye(num_class)[cover_tif]
            one_hot = one_hot.squeeze()
            return one_hot
        
        elif isinstance(cover_tif, torch.Tensor):
        # PyTorch
            if cover_tif.shape[-1] == 1:
                cover_squeezed = cover_tif.squeeze(-1)
            else:
                cover_squeezed = cover_tif         

            eye = torch.eye(num_class, dtype=torch.float32)           
            one_hot = eye[cover_squeezed.long()]
            
            return one_hot
        



if __name__ == '__main__':
    ds= Dataset(root_dir='Sorted_Sources', sources=['rgb', 'hs', 'dtm','sar','lc'], split='train',pca=False, trans=None)
    data= ds[0]
    print(data[0][4].shape)  # Affiche la forme de la première entrée
    lc= Dataset.onehot(ds,data[0][4])
    print(lc.shape)  # Affiche la forme de la première entrée