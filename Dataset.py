import os
import torch
import pandas as pd
import numpy as np
import torch.utils.data as data
import rasterio as rs


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
        # defines sources
        sources = ['Sources/RGB', 'Sources/HS', 'Sources/DEM', 'Labeling/Landuse/tif', 'Labeling/Agriculture/tif'] # TODO change
        # load them
        imgs = []
        for cur_source in sources:
            if cur_source.split('/')[1] == 'HS':
                # imgs.append(np.load(os.path.join(self.root_dir, cur_source, self.fns.iloc[idx]['fns'].split('.')[0] + '.npy')))
                imgs.append(self.read_tif('/ssd_data/pansh_data/', self.fns.iloc[idx]['fns'])) # read the pansh data TODO change
            else:
                imgs.append(self.read_tif(cur_source, self.fns.iloc[idx]['fns']))

        # make them proper type
        num_modalities = 3
        imgs[:num_modalities] = [torch.from_numpy(cur_img).float() for cur_img in imgs[:num_modalities]]
        imgs[num_modalities:] = [torch.from_numpy(cur_img).long().squeeze(-1) for cur_img in imgs[num_modalities:]]

        if self.trans is not None:
            imgs = self.trans(imgs)

        inputs = imgs[:num_modalities]
        targets = imgs[num_modalities:]

        return inputs, targets, self.fns.iloc[idx]['fns']
    