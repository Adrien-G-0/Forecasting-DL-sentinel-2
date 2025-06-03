import numpy as np
import torch
from torch.nn import Sequential

from NewBase import Base
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pickle
from middle_fusion_ import Middle_fusion_en as mf_
# seed
# import random
# from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import measure_flops



class NewArchitectures(Base):
    def __init__(self, params):
        # init base
        super(NewArchitectures, self).__init__(params)
        
        # reorganize sources values
        source_order = ['rgb', 'hs', 'dem', 'sar','lc','sau']
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
                if source == 'lc':
                    input_channels = input_channels +8#self.conf['num_class_lc']  # should be 8
                if source == 'sau':
                    input_channels = input_channels +10#self.conf['num_class_sau']  # should be 10
        
            # define architecture
            self.net=Unet(input_channels)
            # Initialization_weight TODO


        # middle fusion
        elif self.conf['method'] == 'middle_fusion':
            # TODO change the dimensio embedind to 8 or 16
            sources = self.conf['sources']
            self.fusion_en = mf_(self.conf)
            in_channels_middle_fusion = np.sum(self.conf['conf_'+source]["channels"][-1] for source in sources)  # last channel of each source 
            
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
    
    def forward(self, batch): 

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
            # print("flops:" + str(fwd_flops))
            output = self.net(inp)
            return output
        
        
        elif self.conf['method'] == 'middle_fusion':
            

            inp = self.fusion_en(batch)

            with torch.device("meta"):
                model = self.net
                x = inp

            model_fwd = lambda: model(x)
            fwd_flops = measure_flops(model, model_fwd)
            # print("flops:" + str(fwd_flops))

            output = self.net(inp)
            
            return output     



    def create_transform_function(self, transform_list):
        # create transformation function
        def transform_inputs(inps):
            # create transformation
            sources_possibles = ['rgb', 'hs', 'dem', 'sar','lc','sau', 'ndvi']
            inps_dict = {source: inps[i] for i, source in enumerate(self.conf['sources']+['ndvi'])}  # add ndvi to the inputs dict

            # Checking if all keys have a designated value else 0 TODO can maybe be improve to reduce storage and calculations
            inps_dict = {source: inps_dict.get(source, torch.zeros((1,))) for source in sources_possibles}
            rgb, hs, dem, sar, lc, sau, ndvi = inps_dict['rgb'], inps_dict['hs'], inps_dict['dem'], inps_dict['sar'], inps_dict['lc'],inps_dict['sau'], inps_dict['ndvi']


            normalize_rgb, normalize_hs, normalize_dem, normalize_sar, transforms_augmentation = transform_list
            #no normalization for ndvi because it is already between -1 and 1
            ndvi=ndvi.unsqueeze(2) # so ndvi has the same shape as the others
            # no normalization for lc and sau because it is onehot encoded


            rgb = (rgb.numpy() - self.loaded_min_dict_before_normalization['rgb']) / (self.loaded_max_dict_before_normalization['rgb'] - self.loaded_min_dict_before_normalization['rgb'])
            hs = (hs.numpy() - self.loaded_min_dict_before_normalization['hs']) / (self.loaded_max_dict_before_normalization['hs'] - self.loaded_min_dict_before_normalization['hs'])
            dem = (dem.numpy() - self.loaded_min_dict_before_normalization['dem']) / (self.loaded_max_dict_before_normalization['dem'] - self.loaded_min_dict_before_normalization['dem'])
            sar = (sar.numpy() - self.loaded_min_dict_before_normalization['sar']) / (self.loaded_max_dict_before_normalization['sar'] - self.loaded_min_dict_before_normalization['sar'])
            #no need to normalize the ndvi because it is already between -1 and 1 and lc,sau are onehot encoded
            ndvi = ndvi.numpy()
            lc=lc.numpy()
            sau=sau.numpy()

            rgb = rgb.astype(np.float32)
            hs = hs.astype(np.float32)
            dem = dem.astype(np.float32)
            sar = sar.astype(np.float32)
            ndvi = ndvi.astype(np.float32)
            lc = lc.astype(np.float32)
            sau = sau.astype(np.float32)
            
            rgb = normalize_rgb(image=rgb)['image']
            hs = normalize_hs(image=hs)['image']
            dem = normalize_dem(image=dem)['image']
            sar = normalize_sar(image=sar)['image']


            # initialize the transforms
            transforms = A.Compose([transforms_augmentation], is_check_shapes=False,
                                    additional_targets={'hs': 'image',
                                                        'dem': 'image',
                                                        'sar': 'image',
                                                        'lc': 'image',
                                                        'sau': 'image',
                                                        'ndvi': 'image'})
            # apply the transforms
            sample = transforms(image=rgb,
                                hs=hs,
                                dem=dem,
                                sar=sar,
                                lc=lc,
                                sau=sau,
                                ndvi=ndvi
                                
                                )
            # get images
            rgb = sample['image']
            hs = sample['hs']
            dem = sample['dem']
            sar = sample['sar']
            lc= sample['lc']
            sau = sample['sau']
            ndvi = sample['ndvi']

            outputs_dict = {'rgb': rgb, 'hs': hs, 'dem': dem, 'sar': sar ,'lc': lc, 'sau':sau, 'ndvi': ndvi}
            # get needed output values without the others
            output = list(outputs_dict[source] for source in self.conf['sources']) + [ndvi]
            return output
            

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
        output = self.decoder4(x,residual1)


        output=self.activation(output)  
        return 2*output-1 # Rescale output to [-1, 1] range
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == '__main__':
    # # train or test
    # seed = 42
    # random.seed(seed)
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # seed_everything(seed, workers=True)
    # torch.backends.cudnn.deterministic = True

    # Base.main(NewArchitectures)


    model = NewArchitectures.load_from_checkpoint("checkpoints/early_fusion_sar/version_0/checkpoints/last.ckpt")
    dl = model.test_dataloader()
    dataiter = iter(dl)
    try:
        first_batch = next(dataiter)
        print("Premier batch chargé avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement du premier batch: {e}")
