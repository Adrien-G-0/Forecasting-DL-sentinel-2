import numpy as np
import torch
from torch.nn import Sequential
from time import time

import segmentation_models_pytorch as smp
from NewBase import Base
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pickle
from middle_fusion_ import Middle_fusion_en as mf_
# seed
import random
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import measure_flops




class NewArchitectures(Base):#normalement Base
    def __init__(self, params):
        # init base
        super(NewArchitectures, self).__init__(params)
        
        
        # #part to test without base, juste the architecture
        # super().__init__()
        # self.conf=params['conf']
        """End test"""

        # print(self.conf['sources'])

        # #part to test without base, juste the architecture
        # super().__init__()
        # self.conf=params['conf']
        """End test"""

        # print(self.conf['sources'])
        # reorganize sources values
        source_order = ['rgb', 'hs', 'dem', 'sar']
        ordered_sources = []
        for source in source_order:
            if source in self.conf['sources']:
                ordered_sources.append(source)
        self.conf['sources'] = ordered_sources

        # early fusion
        if self.conf['method'] == 'early_fusion':

            input_channels = 0
            for source in self.conf['sources']:
                if source == 'rgb':
                    input_channels = input_channels + 3
                if source == 'hs':
                    input_channels = input_channels + 182
                if source =='dem':
                    input_channels = input_channels + 1
                if source == 'sar' :
                    input_channels = input_channels +2          
        
            # define architecture
            self.net=Unet(input_channels)
            # Initialization_weight TODO


        # middle fusion
        elif self.conf['method'] == 'middle_fusion':
            # TODO
            sources = self.conf['sources']
            self.fusion_en = mf_(sources)
            in_channels_middle_fusion = len(sources) * 64
            
            
            
            
            # define architecture
            self.net = Unet(in_channels_middle_fusion)
            # Initialization_weight TODO


        self.mean_dict = self.load_dict(self.conf['mean_dict_01'])
        self.std_dict = self.load_dict(self.conf['std_dict_01'])
        self.max_dict = self.load_dict(self.conf['max_dict_01'])
        self.loaded_min_dict_before_normalization = self.load_dict(self.conf['min_dict'])
        self.loaded_max_dict_before_normalization = self.load_dict(self.conf['max_dict'])

    def load_dict(self, name):
        with open(name, 'rb') as f:
            loaded_dict = pickle.load(f)

        return loaded_dict
    
    def forward(self, batch): # identical as the KnownArchitecture but
        # with the new U-net 

        components = {}
        for source, input_tensor in zip(self.conf['sources'], batch):
            components[source] = input_tensor
            
        
        if self.conf['method'] == 'early_fusion':
            first_flag = True
            inp = None
        
            for source in self.conf['sources']:
                if first_flag:
                    inp = components[source]
                    first_flag = False
                else:
                    print(inp.shape, components[source].shape)
                    inp = torch.cat([inp, components[source]], axis=1)

            with torch.device("meta"):
                model = self.net
                x = inp

            model_fwd = lambda: model(x)
            fwd_flops = measure_flops(model, model_fwd)
            print("flops:" + str(fwd_flops))

            # apply
            return self.net(inp)
        
        
        elif self.conf['method'] == 'middle_fusion':
            # middle fusion TODO change to use a better version that can manage all the cases at once

            inp = self.fusion_en(batch)

            with torch.device("meta"):
                model = self.net
                x = inp

            model_fwd = lambda: model(x)
            fwd_flops = measure_flops(model, model_fwd)
            print("flops:" + str(fwd_flops))

            return self.net(inp)       



    def create_transform_function(self, transform_list):
        # create function
        def transform_inputs(inps):
            # create transformation
            rgb, hs, dem, sar, ndvi = inps
            normalize_rgb, normalize_hs, normalize_dem, normalize_sar, transforms_augmentation = transform_list
            print("Structure de transform_list:", type(transform_list), transform_list)
            
            # ipdb.set_trace()
            transforms = A.Compose([transforms_augmentation], is_check_shapes=False,
                                    additional_targets={'hs': 'image',
                                                        'dem': 'image',
                                                        'sar': 'image',
                                                        'ndvi': 'image'})
            
            rgb = (rgb.permute(1,2,0).numpy() - self.loaded_min_dict_before_normalization['rgb']) / (self.loaded_max_dict_before_normalization['rgb'] - self.loaded_min_dict_before_normalization['rgb'])
            hs = (hs.permute(1,2,0).numpy() - self.loaded_min_dict_before_normalization['hs']) / (self.loaded_max_dict_before_normalization['hs'] - self.loaded_min_dict_before_normalization['hs'])
            dem = (dem.permute(1,2,0).numpy() - self.loaded_min_dict_before_normalization['dem']) / (self.loaded_max_dict_before_normalization['dem'] - self.loaded_min_dict_before_normalization['dem'])
            sar = (sar.permute(1,2,0).numpy() - self.loaded_min_dict_before_normalization['sar']) / (self.loaded_max_dict_before_normalization['sar'] - self.loaded_min_dict_before_normalization['sar'])
            #no need to normalize the ndvi because it is already between -1 and 1
            if ndvi.dim() == 2:  # Si c'est déjà un format HxW
                ndvi_np = ndvi.numpy()
            else:  # Si c'est un format CxHxW avec C=1
                ndvi_np = ndvi.squeeze(0).numpy()  # Enlever la dimension du canal et convertir en numpy
            
            # Ajouter une dimension pour la compatibilité avec Albumentations (qui attend HxWxC)
            ndvi = np.expand_dims(ndvi_np, axis=2).astype(np.float32)
            

            rgb = rgb.astype(np.float32)
            hs = hs.astype(np.float32)
            dem = dem.astype(np.float32)
            sar = sar.astype(np.float32)
            ndvi = ndvi.astype(np.float32) if torch.is_tensor(ndvi) else ndvi.astype(np.float32)
            
            rgb = normalize_rgb(image=rgb)['image']
            hs = normalize_hs(image=hs)['image']
            dem = normalize_dem(image=dem)['image']
            sar = normalize_sar(image=sar)['image']


            # TODO how to modify this part? 
            sample = transforms(image=rgb,
                                hs=hs,
                                dem=dem,
                                sar=sar,
                                ndvi=ndvi
                                
                                )
            # get images
            rgb = sample['image']
            hs = sample['hs']
            dem = sample['dem']
            sar = sample['sar']
            ndvi = sample['ndvi']

            # return results
            return rgb, hs, dem, sar, ndvi

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
        




###### Creation of the classes for the Unet ######

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn=torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block1 = ConvBlock(in_channels, out_channels)
        self.conv_block2 = ConvBlock(out_channels, out_channels)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        residual = x
        output = self.pool(x)
        return output, residual
        
class UpsamplingBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, skip_channels):
            super(UpsamplingBlock, self).__init__()
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_block1 = ConvBlock(in_channels, in_channels)
            self.conv_block2 = ConvBlock(in_channels + skip_channels, in_channels)
            self.conv_block3 = ConvBlock(in_channels, out_channels)

        def forward(self, x, skip_connection):
            x = self.upsample(x)
            x = self.conv_block1(x)
            
            x = torch.cat((x, skip_connection), dim=1)
            
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            return x


class Unet(torch.nn.Module):
    def __init__(self,input_channels):
        super().__init__()
        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.bottleneck = Sequential(ConvBlock(512, 1024), ConvBlock(1024, 512))
        
        self.decoder1 = UpsamplingBlock(512, 256, 512)
        self.decoder2 = UpsamplingBlock(256, 128, 256)
        self.decoder3 = UpsamplingBlock(128, 64, 128)
        self.decoder4 = UpsamplingBlock(64, 1, 64)

        # self.final_conv = torch.nn.Conv2d(64, 1, kernel_size=1) # opération similaire à une couche linéaire mais mieux car préserve l'aspect spatial
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x, residual1 = self.encoder1(x)
        x, residual2 = self.encoder2(x)
        x, residual3 = self.encoder3(x)
        x, residual4 = self.encoder4(x)

        x = self.bottleneck(x)

        x = self.decoder1(x,residual4)
        x = self.decoder2(x,residual3)
        x = self.decoder3(x,residual2)
        x = self.decoder4(x,residual1)

        # x = self.final_conv(x) #ajout par rapport au modèle car les dimension ne correspondent pas
        output=self.activation(x)  # sigmoïd that return a number between 0 and 1
        return 2*output-1 # to return NDVI value between -1 and 1
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == '__main__':
    # deb = time()
    # source=['rgb', 'hs', 'dem','sar']
    # model=mf_(source)
    # print(model)
    # inputa_rgb=torch.randn(1,3,256,256)
    # inputa_hs=torch.randn(1,182,256,256)
    # inputa_dem=torch.randn(1,1,256,256)
    # inputa_sar=torch.randn(1,2,256,256)
    # inputs=[inputa_rgb, inputa_hs,inputa_dem, inputa_sar]
    # output=model(inputs)
    # print(output.shape)
    # fin = time()
    # print("Time:", fin-deb)
    

    # train or test
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True

    Base.main(NewArchitectures)

    
