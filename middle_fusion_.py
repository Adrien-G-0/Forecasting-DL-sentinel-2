import torch
import torch.nn as nn

class ConvBlock_mf(nn.Module):
    def __init__(self, conf):
        super(ConvBlock_mf, self).__init__()
        channels = conf['channels']
        kernels = conf['kernels']
        
        layers = []
        for i in range(len(kernels)):
            layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=kernels[i], padding=kernels[i]//2,stride=1))
            layers.append(nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)

class Middle_fusion_en(nn.Module):
    def __init__(self, conf):
        super(Middle_fusion_en, self).__init__()
        self.sources = conf["sources"]
        self.conv = nn.ModuleDict()
        
        # Store configurations
        self.conf_rgb = conf["conf_rgb"]
        self.conf_dem = conf["conf_dem"]
        self.conf_sar = conf["conf_sar"]
        self.conf_hs = conf["conf_hs"]
        self.conf_lc = conf["conf_lc"]
        self.conf_sau = conf["conf_sau"]
        
        # Check if the sources are valid
        if not all(source in ['rgb', 'hs',  'dem','sar','lc','sau'] for source in self.sources):
            raise Exception("Invalid sources, must be in ['rgb', 'hs', 'dem', 'sar','lc','sau' ]")
        
        # Check if the number of channels and kernels are consistent
        if len(self.conf_rgb['channels']) != len(self.conf_rgb['kernels']) + 1:
            raise Exception("RGB configurations are wrong, channels length must be equal to kernels length + 1")
        if len(self.conf_hs['channels']) != len(self.conf_hs['kernels']) + 1:
            raise Exception("HS configurations are wrong, channels length must be equal to kernels length + 1")
        if len(self.conf_sar['channels']) != len(self.conf_sar['kernels']) + 1:
            raise Exception("SAR configurations are wrong, channels length must be equal to kernels length + 1")
        if len(self.conf_dem['channels']) != len(self.conf_dem['kernels']) + 1:
            raise Exception("DEM configurations are wrong, channels length must be equal to kernels length + 1")
        if len(self.conf_lc['channels']) != len(self.conf_lc['kernels']) + 1:
            raise Exception("LC configurations are wrong, channels length must be equal to kernels length + 1")
        if len(self.conf_sau['channels']) != len(self.conf_sau['kernels']) + 1: 
            raise Exception("SAU configurations are wrong, channels length must be equal to kernels length + 1")
        
        # Create convolutional blocks for each source
        for source in self.sources:
            if source == 'rgb':
                self.conv[source] = ConvBlock_mf(self.conf_rgb)
            elif source == 'hs':
                self.conv[source] = ConvBlock_mf(self.conf_hs)
            elif source == 'sar':
                self.conv[source] = ConvBlock_mf(self.conf_sar)
            elif source == 'dem':
                self.conv[source] = ConvBlock_mf(self.conf_dem)
            elif source == 'lc':
                self.conv[source] = ConvBlock_mf(self.conf_lc)
            elif source == 'sau':
                self.conv[source] = ConvBlock_mf(self.conf_sau)
        
        # # Register the modules properly
        # for source in self.sources:
        #     setattr(self, f'conv_{source}', self.conv[source])
        
        
        # for source in self.sources:
        #    config = getattr(self, f'conf_{source}', None)
        #    self.conv[source] = ConvBlock_mf(config)


    
        
    def forward(self, inputs):
        # inputs is a list or tuple of tensors in the same order as self.sources
        if len(inputs) != len(self.sources):
            raise Exception(f"Expected {len(self.sources)} inputs, got {len(inputs)}")
        
        features = []
        # Apply the corresponding convolutional block to each input 
        for i, source in enumerate(self.sources):
            feature = self.conv[source](inputs[i])
            features.append(feature)
        
        return torch.cat(features, dim=1)
    



if __name__ == '__main__':
        

#  conf_rgb={'channels':[3,16,32,64], 'kernels':[3,3,3]},
#  conf_hs={'channels':[182,128,64], 'kernels':[3,3]},
#  conf_dem={'channels':[1,16,32,64], 'kernels':[3,3,3]},
#  conf_sar={'channels':[2,16,32,64], 'kernels':[3,3,3]},
#  conf_lc={'channels':[8,32,64], 'kernels':[3,3]},
#  conf_sau={'channels':[10,32,64], 'kernels':[3,3]}
        import json
        with open('params.json') as f:
            conf = json.load(f)
        model=Middle_fusion_en(conf)
        print(model)
        inputa_rgb=torch.randn(1,3,256,256)
        inputa_hs=torch.randn(1,182,256,256)
        inputa_dem=torch.randn(1,1,256,256)
        inputa_sar=torch.randn(1,2,256,256)
        inputa_lc=torch.randn(1,8,256,256)
        inputa_sau=torch.randn(1,10,256,256)
        inputs=[inputa_dem, inputa_sar,inputa_lc,inputa_sau]
        output=model(inputs)
        print(output.shape)