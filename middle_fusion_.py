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
    def __init__(self, sources,
                conf_rgb={'channels':[3,16,32,64], 'kernels':[3,3,3]},
                conf_hs={'channels':[182,128,64], 'kernels':[3,3]},
                conf_dem={'channels':[1,16,32,64], 'kernels':[3,3,3]},
                conf_sar={'channels':[2,16,32,64], 'kernels':[3,3,3]},
                conf_lc={'channels':[8,32,64], 'kernels':[3,3]},
                ):
        super(Middle_fusion_en, self).__init__()
        self.sources = sources
        self.conv = {}
        
        # Store configurations
        self.conf_rgb = conf_rgb
        self.conf_dem = conf_dem
        self.conf_sar = conf_sar
        self.conf_hs = conf_hs
        self.conf_lc = conf_lc
        
        # Check if the sources are valid
        if not all(source in ['rgb', 'hs',  'dem','sar','lc'] for source in self.sources):
            raise Exception("Invalid sources, must be in ['rgb', 'hs', 'dem', 'sar','lc' ]")
        
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
        
        # Register the modules properly
        for source in self.sources:
            setattr(self, f'conv_{source}', self.conv[source])
    


    
        
    def forward(self, inputs):
        # inputs is a list or tuple of tensors in the same order as self.sources
        if len(inputs) != len(self.sources):
            raise Exception(f"Expected {len(self.sources)} inputs, got {len(inputs)}")
        
        features = []
        for i, source in enumerate(self.sources):
            feature = self.conv[source](inputs[i])
            features.append(feature)
        
        return torch.cat(features, dim=1)
    



if __name__ == '__main__':
        source=['rgb', 'hs', 'dem','sar','lc']
        model=Middle_fusion_en(source)
        print(model)
        inputa_rgb=torch.randn(1,3,256,256)
        inputa_hs=torch.randn(1,182,256,256)
        inputa_dem=torch.randn(1,1,256,256)
        inputa_sar=torch.randn(1,2,256,256)
        inputa_lc=torch.randn(1,7,256,256)
        inputs=[inputa_rgb, inputa_hs,inputa_dem, inputa_sar,inputa_lc]
        output=model(inputs)
        print(output.shape)