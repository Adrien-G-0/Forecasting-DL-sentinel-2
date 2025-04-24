import torch
import torch.nn as nn

class ConvBlock_mf(nn.Module):
    def __init__(self, conf):
        super(ConvBlock_mf, self).__init__()
        channels = conf['channels']
        kernels = conf['kernels']
        
        layers = []
        for i in range(len(kernels)):
            layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=kernels[i], padding=kernels[i]//2),stride=1)
            layers.append(nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)

class Middle_fusion_en(nn.Module):
    def __init__(self, sources,
                conf_rgb={'channels':[3,16,32,64], 'kernels':[3,3,3]},
                conf_dem={'channels':[1,16,32,64], 'kernels':[3,3,3]},
                conf_sar={'channels':[2,16,32,64], 'kernels':[3,3,3]},
                conf_hs={'channels':[182,128,64], 'kernels':[3,3]}
                ):
        super(Middle_fusion_en, self).__init__()
        self.sources = sources
        self.conv = {}
        
        # Store configurations
        self.conf_rgb = conf_rgb
        self.conf_dem = conf_dem
        self.conf_sar = conf_sar
        self.conf_hs = conf_hs
        
        # Check if the sources are valid
        if not all(source in ['rgb', 'hs', 'sar', 'dem'] for source in self.sources):
            raise Exception("Invalid sources, must be in ['rgb', 'hs', 'sar', 'dem']")
        
        # Check if the number of channels and kernels are consistent
        if len(self.conf_rgb['channels']) != len(self.conf_rgb['kernels']) + 1:
            raise Exception("RGB configurations are wrong, channels length must be equal to kernels length + 1")
        if len(self.conf_hs['channels']) != len(self.conf_hs['kernels']) + 1:
            raise Exception("HS configurations are wrong, channels length must be equal to kernels length + 1")
        if len(self.conf_sar['channels']) != len(self.conf_sar['kernels']) + 1:
            raise Exception("SAR configurations are wrong, channels length must be equal to kernels length + 1")
        if len(self.conf_dem['channels']) != len(self.conf_dem['kernels']) + 1:
            raise Exception("DEM configurations are wrong, channels length must be equal to kernels length + 1")
        
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
        
        # Register the modules properly
        for source in self.sources:
            setattr(self, f'conv_{source}', self.conv[source])
    
    def forward(self, inputs):
        # inputs should be a dictionary with keys being the source names
        # and values being the corresponding input tensors
        features = []
        
        for source in self.sources:
            if source in inputs:
                # Process each input with its corresponding convolutional block
                feature = self.conv[source](inputs[source])
                features.append(feature)
        
        # Concatenate all features along the channel dimension (dim=1)
        if features:
            return torch.cat(features, dim=1)
        else:
            raise Exception("No valid inputs provided for the available sources")
        
    def forward(self, inputs):
        # inputs is a list or tuple of tensors in the same order as self.sources
        if len(inputs) != len(self.sources):
            raise Exception(f"Expected {len(self.sources)} inputs, got {len(inputs)}")
        
        features = []
        for i, source in enumerate(self.sources):
            feature = self.conv[source](inputs[i])
            features.append(feature)
        
        return torch.cat(features, dim=1)