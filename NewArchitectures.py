import numpy as np
import torch
from torch.nn import Sequential
from time import time

import segmentation_models_pytorch as smp
from Base import Base
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pickle
from middle_fusion_ import Middle_fusion_en as mf_
# seed
import random
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import measure_flops




class NewArchitectures(Base):
    def __init__(self, params):
        # init base
        super(NewArchitectures, self).__init__(params)

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
                if source =='dem':
                    input_channels = input_channels + 1
                if source == 'hs':
                    input_channels = input_channels + 182
                if source == 'sar' :
                    input_channels = input_channels +2          
        
            # define architecture
            self.net=Unet(input_channels)
            # Initialization_weight TODO


        # middle fusion
        #TODO ajouter les valeurs de SAR data dans ledans chaques cas
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
    deb = time()
    source=['rgb', 'hs', 'dem','sar']
    model=mf_(source)
    print(model)
    inputa_rgb=torch.randn(1,3,256,256)
    inputa_hs=torch.randn(1,182,256,256)
    inputa_dem=torch.randn(1,1,256,256)
    inputa_sar=torch.randn(1,2,256,256)
    inputs=[inputa_rgb, inputa_hs,inputa_dem, inputa_sar]
    output=model(inputs)
    print(output.shape)

    input_channels = output.shape[1]
    print("input_channels:", input_channels)
    input_tensor = torch.randn(1, input_channels, 256, 256)
    model = Unet(input_channels=input_channels)

    print("=== ENCODING ===")
    x, residual1 = model.encoder1(input_tensor)
    print("encoder1 -> x:", x.size(), "| residual1:", residual1.size())

    x, residual2 = model.encoder2(x)
    print("encoder2 -> x:", x.size(), "| residual2:", residual2.size())

    x, residual3 = model.encoder3(x)
    print("encoder3 -> x:", x.size(), "| residual3:", residual3.size())

    x, residual4 = model.encoder4(x)
    print("encoder4 -> x:", x.size(), "| residual4:", residual4.size())

    print("\n=== BOTTLENECK ===")
    x = model.bottleneck(x)
    print("bottleneck -> x:", x.size())

    print("\n=== DECODING ===")
    x = model.decoder1(x, residual4)
    print("decoder1 -> x:", x.size())

    x = model.decoder2(x, residual3)
    print("decoder2 -> x:", x.size())

    x = model.decoder3(x, residual2)
    print("decoder3 -> x:", x.size())

    x = model.decoder4(x, residual1)
    print("decoder4 -> x:", x.size())

    print("\n=== OUTPUT ===")
    output = model.activation(x)
    print("input_tensor:", input_tensor.size())
    print("output:", output.size())

    print('\nTemps pour un passage dans le modèle pour une image :', round(time()-deb,3),'s')
        
    num_params = count_parameters(model)
    print(f"Le modèle a {num_params} paramètres.")
    

    
