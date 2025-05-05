import numpy as np
import torch
from einops import rearrange, repeat # , reduce
import segmentation_models_pytorch as smp
from Base import Base
import albumentations as A #data augmentation library.
from albumentations.pytorch.transforms import ToTensorV2
import pickle
from middle_fusion_rgb_hs import Middle_fusion_en as mf_rgb_hs
from middle_fusion_rgb_dem import Middle_fusion_en as mf_rgb_dem
from middle_fusion_rgb_hs_dem import Middle_fusion_en as mf_rgb_hs_dem
# TODO add 
from middle_fusion_rgb_sar import Middle_fusion_en as mf_rgb_sar
from middle_fusion_rgb_hs_sar import Middle_fusion_en as mf_rgb_hs_sar
from middle_fusion_sar_hs import Middle_fusion_en as mf_sar_hs
from middle_fusion_hs_dem_sar import Middle_fusion_en as mf_hs_dem_sar
from middle_fusion_rgb_dem_sar import Middle_fusion_en as mf_rgb_dem_sar
# seed random.seed(seed), torch.manual_seed(seed), np.random.seed(seed): 
# Ensures the same random numbers are generated across different runs for reproducibility.
import random 
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import measure_flops

import glob

class KnownArchitectures(Base):
    def __init__(self, params):
        # init base
        super(KnownArchitectures, self).__init__(params)
        #Calls the parent constructor super().__init__(params), 
        # initializing the base functionalities

        # early fusion
        if self.conf['method'] == 'early_fusion':

            input_channels = 0
            for source in self.conf['sources']:
                if source == 'rgb':
                    input_channels = input_channels + 3
                if source == 'dtm':
                    input_channels = input_channels + 1
                if source == 'hs':
                    input_channels = input_channels + 182
                if source == 'sar':
                    input_channels = input_channels + 2

            # define architecture
            # Defines a U-Net segmentation model using segmentation_models_pytorch (smp).
            self.net = smp.Unet(
                encoder_name=self.conf['encoder_name'],
                encoder_weights= (self.conf['encoder_weights'] if self.conf['encoder_weights'] != "None" else None),
                in_channels=input_channels,
                classes=self.conf['n_classes_landuse'] + self.conf['n_classes_agricolture'],
            )
            
            # Applies Xavier uniform initialization to the first convolutional layer.
            torch.nn.init.xavier_uniform_(self.net.encoder.conv1.weight) # reinitialize first layer

        # middle fusion
        elif self.conf['method'] == 'middle_fusion':
            if 'rgb' in self.conf['sources'] and 'hs' in self.conf['sources']:
                self.fusion_en = mf_rgb_hs(conf_rgb={'channels':[3,16,32,64], 'kernels':[3,3,3]},
                                        conf_hs={'channels':[182,128,64], 'kernels':[3,3]})
                in_channels_middle_fusion = 64+64

                # conf_rgb={'channels':[3,16,32,64], 'kernels':[3,3,3]}
                # RGB has 3 input channels (Red, Green, Blue).
                # Passes through 3 convolutional layers with increasing feature channels: 16 → 32 → 64.
                # Uses 3×3 kernels for all layers.                                                                                                  

            elif 'rgb' in self.conf['sources'] and 'hs' in self.conf['sources'] and 'dtm' in self.conf['sources']:
                self.fusion_en = mf_rgb_hs_dem(conf_rgb={'channels':[3,16,32,64], 'kernels':[3,3,3]},
                                            conf_hs={'channels':[182,128,64], 'kernels':[3,3]},
                                            conf_dem={'channels':[1,16,32,64], 'kernels':[3,3,3]})
                in_channels_middle_fusion = 64+64+64

                # conf_hs={'channels':[182,128,64], 'kernels':[3,3]}
                # Hyperspectral (HS) has 182 input channels.
                # Passes through 2 convolutional layers with decreasing channels: 128 → 64.
                # Uses 3×3 kernels.

            elif 'rgb' in self.conf['sources'] and 'dtm' in self.conf['sources']:
                self.fusion_en = mf_rgb_dem(conf_rgb={'channels':[3,16,32,64], 'kernels':[3,3,3]},
                                            conf_dem={'channels':[1,16,32,64], 'kernels':[3,3,3]})
                in_channels_middle_fusion = 64+64
            
            # TODO create here middle fusion module and if section to get SAR data            
            elif 'rgb' in self.conf['sources'] and 'sar' in self.conf['sources']:
                self.fusion_en = mf_rgb_sar(conf_rgb={'channels':[3,16,32,64], 'kernels':[3,3,3]},
                                            conf_sar={'channels':[2,16,32,64], 'kernels':[3,3,3]})
                in_channels_middle_fusion = 64+64  

            elif 'rgb' in self.conf['sources'] and 'hs' in self.conf['sources'] and 'sar' in self.conf['sources']:
                self.fusion_en = mf_rgb_hs_sar(conf_rgb={'channels':[3,16,32,64], 'kernels':[3,3,3]},
                                            conf_hs={'channels':[182,128,64], 'kernels':[3,3]},
                                            conf_sar={'channels':[2,16,32,64], 'kernels':[3,3,3]})
                in_channels_middle_fusion = 64+64+64

            elif 'sar' in self.conf['sources'] and 'hs' in self.conf['sources']:
                self.fusion_en = mf_sar_hs(conf_sar={'channels':[2,16,32,64], 'kernels':[3,3,3]},
                                           conf_hs={'channels':[182,128,64], 'kernels':[3,3]})
                in_channels_middle_fusion = 64+64  

            elif 'dtm' in self.conf['sources'] and 'hs' in self.conf['sources'] and 'sar' in self.conf['sources']:
                self.fusion_en = mf_hs_dem_sar(conf_dem={'channels':[1,16,32,64], 'kernels':[3,3,3]},
                                   conf_hs={'channels':[182,128,64], 'kernels':[3,3]},
                                   conf_sar={'channels':[2,16,32,64], 'kernels':[3,3,3]})
                in_channels_middle_fusion = 64+64+64

            elif 'rgb' in self.conf['sources'] and 'dtm' in self.conf['sources'] and 'sar' in self.conf['sources']:
                self.fusion_en = mf_rgb_dem_sar(conf_rgb={'channels':[3,16,32,64], 'kernels':[3,3,3]},
                                    conf_dem={'channels':[1,16,32,64], 'kernels':[3,3,3]},
                                    conf_sar={'channels':[2,16,32,64], 'kernels':[3,3,3]})
                in_channels_middle_fusion = 64+64+64


            # define architecture
            self.net = smp.Unet(
                encoder_name=self.conf['encoder_name'],
                encoder_weights= (self.conf['encoder_weights'] if self.conf['encoder_weights'] != "None" else None),
                in_channels=in_channels_middle_fusion,
                classes=self.conf['n_classes_landuse'] + self.conf['n_classes_agricolture'],
            )
            
            torch.nn.init.xavier_uniform_(self.net.encoder.conv1.weight) # reinitialize first layer
 
        
        # TODO
        #Loading Precomputed Mean, Std, Max, Min Values, in this part, need to load SAR mean, max, etc.    
        self.mean_dict = self.load_dict(self.conf['mean_dict_01'])
        self.std_dict = self.load_dict(self.conf['std_dict_01'])
        self.max_dict = self.load_dict(self.conf['max_dict_01'])
        #these 2 lines are without norm
        self.loaded_min_dict_before_normalization = self.load_dict(self.conf['min_dict'])
        self.loaded_max_dict_before_normalization = self.load_dict(self.conf['max_dict'])


    def load_dict(self, name):
        with open(name, 'rb') as f:
            loaded_dict = pickle.load(f)

        return loaded_dict
        
    # Original forward
    def forward(self, batch):

        rgb, hs, dtm, sar = batch

        if self.conf['method'] == 'early_fusion':
            first_flag = True
            if 'rgb' in self.conf['sources']:
                if first_flag:
                    inp = rgb
                    first_flag = False
                else:
                    inp = torch.cat([inp, rgb], axis=1)

            if 'hs' in self.conf['sources']:
                if first_flag:
                    inp = hs
                    first_flag = False
                else:
                    inp = torch.cat([inp, hs], axis=1)

            if 'dtm' in self.conf['sources']:
                if first_flag:
                    inp = dtm
                    first_flag = False
                else:
                    inp = torch.cat([inp, dtm], axis=1)

            # TODO add sar here
            if 'sar' in self.conf['sources']: 
                if first_flag:
                    inp = sar
                    first_flag = False
                else:
                    inp = torch.cat([inp, sar], axis=1)


            # with torch.device("meta"):
            #     model = self.net
            #     x = inp

            # model_fwd = lambda: model(x)
            # fwd_flops = measure_flops(model, model_fwd)
            # print("flops:" + str(fwd_flops))

            # apply
            return self.net(inp)
        
        # middle fusion
        elif self.conf['method'] == 'middle_fusion':
            if 'rgb' in self.conf['sources'] and 'hs' in self.conf['sources']:
                inp = rgb, hs
            elif 'rgb' in self.conf['sources'] and 'hs' in self.conf['sources'] and 'dtm' in self.conf['sources']:
                inp = rgb, hs, dtm
            elif 'rgb' in self.conf['sources'] and 'dtm' in self.conf['sources']:
                inp = rgb, dtm
            elif 'rgb' in self.conf['sources'] and 'sar' in self.conf['sources']:
                inp = rgb, sar
            elif 'rgb' in self.conf['sources'] and 'hs' in self.conf['sources'] and 'sar' in self.conf['sources']:
                inp = rgb, hs, sar
            elif 'sar' in self.conf['sources'] and 'hs' in self.conf['sources']:
                inp = sar, hs
            
            inp = self.fusion_en(inp)

            with torch.device("meta"):
                model = self.net
                x = inp

            # model_fwd = lambda: model(x)
            # fwd_flops = measure_flops(model, model_fwd)
            # print("flops:" + str(fwd_flops))

            return self.net(inp)
            

    def create_transform_function(self, transform_list):
        # create function
        def transform_inputs(inps):
            # create transformation

            rgb, hs, dem, sar, gt_lu, gt_ag = inps
            normalize_rgb, normalize_hs, normalize_dem, normalize_sar, transforms_augmentation = transform_list

            # ipdb.set_trace()
            transforms = A.Compose([transforms_augmentation], is_check_shapes=False,
                                    additional_targets={'hs': 'image',
                                                        'dem': 'image',
                                                        'sar': 'image',
                                                        'gt_ag': 'mask',}) #why only gt_ag, there is no another gt?

            rgb = (rgb.permute(1,2,0).numpy() - self.loaded_min_dict_before_normalization['rgb']) / (self.loaded_max_dict_before_normalization['rgb'] - self.loaded_min_dict_before_normalization['rgb'])
            hs = (hs.permute(1,2,0).numpy() - self.loaded_min_dict_before_normalization['hs']) / (self.loaded_max_dict_before_normalization['hs'] - self.loaded_min_dict_before_normalization['hs'])
            dem = (dem.permute(1,2,0).numpy() - self.loaded_min_dict_before_normalization['dem']) / (self.loaded_max_dict_before_normalization['dem'] - self.loaded_min_dict_before_normalization['dem'])
            sar = (sar.permute(1,2,0).numpy() - self.loaded_min_dict_before_normalization['sar']) / (self.loaded_max_dict_before_normalization['sar'] - self.loaded_min_dict_before_normalization['sar'])

            rgb = normalize_rgb(image=rgb)['image']
            hs = normalize_hs(image=hs)['image']
            dem = normalize_dem(image=dem)['image']
            sar = normalize_sar(image=sar)['image']


            # TODO how to modify this part? 
            sample = transforms(image=rgb,
                                mask=gt_lu.permute(1,2,0).numpy(),
                                hs=hs,
                                dem=dem,
                                sar=sar,
                                gt_ag=gt_ag.permute(1,2,0).numpy()
                                )
            
            # get images
            rgb = sample['image']
            gt_lu = sample['mask'].long().permute(2,0,1).squeeze(dim=0)
            gt_ag = sample['gt_ag'].long().permute(2,0,1).squeeze(dim=0)
            hs = sample['hs']
            dem = sample['dem']
            sar = sample['sar']

            # return results
            return rgb, hs, dem, sar, gt_lu, gt_ag # Change back

        # return the function
        return transform_inputs

    def train_transforms(self):
        # define training size
        train_size = self.conf['train_size'] if 'train_size' in self.conf else self.conf['input_size']
        # create transformation

        normalize_rgb = A.Normalize(mean=self.mean_dict['rgb'], std=self.std_dict['rgb'], max_pixel_value=self.max_dict['rgb'])
        normalize_hs = A.Normalize(mean=self.mean_dict['hs'], std=self.std_dict['hs'], max_pixel_value=self.max_dict['hs'])
        normalize_dem = A.Normalize(mean=self.mean_dict['dem'], std=self.std_dict['dem'], max_pixel_value=self.max_dict['dem'])
        normalize_sar = A.Normalize(mean=self.mean_dict['sar'], std=self.std_dict['sar'], max_pixel_value=self.max_dict['sar'])

        transforms_augmentation = A.Compose([A.Resize(*self.conf['input_size']),
            A.crops.transforms.RandomCrop(*train_size),
            A.Rotate(limit=[-180, 180]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            ToTensorV2()
        ], is_check_shapes=False)

        transforms = normalize_rgb, normalize_hs, normalize_dem, normalize_sar, transforms_augmentation

        # create transform function
        return self.create_transform_function(transforms)
        


    def val_transforms(self):
        print("self.conf keys:", self.conf.keys())
        print("self.mean_dict:", self.mean_dict if hasattr(self, 'mean_dict') else "Not Defined")
        print("self.std_dict:", self.std_dict if hasattr(self, 'std_dict') else "Not Defined")
        print("self.max_dict:", self.max_dict if hasattr(self, 'max_dict') else "Not Defined")
        normalize_rgb = A.Normalize(
            mean=self.mean_dict['rgb'], 
            std=self.std_dict['rgb'], 
            max_pixel_value=self.max_dict['rgb']
        )

        # create transformation
        normalize_rgb = A.Normalize(mean=self.mean_dict['rgb'], std=self.std_dict['rgb'], max_pixel_value=self.max_dict['rgb'])
        normalize_hs = A.Normalize(mean=self.mean_dict['hs'], std=self.std_dict['hs'], max_pixel_value=self.max_dict['hs'])
        normalize_dem = A.Normalize(mean=self.mean_dict['dem'], std=self.std_dict['dem'], max_pixel_value=self.max_dict['dem'])
        normalize_sar = A.Normalize(mean=self.mean_dict['sar'], std=self.std_dict['sar'], max_pixel_value=self.max_dict['sar'])

        transforms_augmentation = A.Compose([
            A.Resize(*self.conf['input_size']),
            ToTensorV2()
        ], is_check_shapes=False)

        transforms = normalize_rgb, normalize_hs, normalize_dem, normalize_sar, transforms_augmentation
    
        # create transform function
        return self.create_transform_function(transforms)
    
    def test_transforms(self):
        return self.val_transforms()
        

if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    # train or test
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True

    Base.main(KnownArchitectures)
        