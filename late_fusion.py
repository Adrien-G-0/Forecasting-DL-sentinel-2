import torch
from torch import nn
import numpy as np

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

class LateFusionEncoder(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.sources=conf["sources"]
        self.dict_module={}

        for source in self.sources:
            input_channels=conf["conf_"+source]['channels'][0]
            self.dict_module[source] = nn.ModuleList([
                EncoderBlock(input_channels, 2*input_channels),
                EncoderBlock(2*input_channels, 4*input_channels),
                EncoderBlock(4*input_channels, 8*input_channels),
                EncoderBlock(8*input_channels, 16*input_channels)
            ])

    def forward(self, inputs):
        '''
        Residual are returned in by deepth of creation, the first element is from the first layer
        '''
        if len(inputs) != len(self.sources):
            raise Exception(f"Expected {len(self.sources)} inputs, got {len(inputs)}")
        
        dict_residual={}
        for i,source in enumerate(self.sources):
            dict_residual[source] = []
            inp=inputs[i]
            dict_residual[source]=[]
            for module in self.dict_module[source]:
                inp, residual = module(inp)
                dict_residual[source].append(residual)
            
        # Concatenate the residuals from all sources
        skip_connections = [torch.cat([dict_residual[source][i] for source in self.sources], dim=1) 
            for i in range(len(dict_residual[self.sources[0]]))]


        return skip_connections



class LateFusionDecoder(nn.Module):
    def __init__(self,conf):
        super().__init__()
        self.sources = conf["sources"]
        self.input_channels = np.sum([conf["conf_"+source]["channels"][0] for source in self.sources]) *2**4  # Assuming the input channels are doubled at each stage

        self.upsampling_list = nn.ModuleList([
            UpsamplingBlock(self.input_channels, self.input_channels // 2,self.input_channels),
            UpsamplingBlock(self.input_channels // 2, self.input_channels // 4,self.input_channels // 2),
            UpsamplingBlock(self.input_channels // 4, self.input_channels // 8,self.input_channels // 4),
            UpsamplingBlock(self.input_channels // 8, 1,self.input_channels // 8)
        ])

    def forward(self, list_residuals):
        list_residuals = list(reversed(list_residuals))  # Reverse the order of residuals to start from the last layer
        x=torch.zeros_like(list_residuals[0])
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        for res,decoder in zip(list_residuals,self.upsampling_list):
            x = decoder(x, res)
        outputs = x
            
        return outputs



class LateFusion(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.sources = conf["sources"]
        self.encoder = LateFusionEncoder(conf)
        self.decoder = LateFusionDecoder(conf)
        self.activation= nn.Sigmoid()

    def forward(self, inputs):
        if len(inputs) != len(self.sources):
            raise Exception(f"Expected {len(self.sources)} inputs, got {len(inputs)}")
        
        list_residuals = self.encoder(inputs)
        outputs = self.decoder(list_residuals)
        output= 2* self.activation(outputs)-1

        return output
    


if __name__ == '__main__':
    import json
    with open('params.json') as f:
        conf = json.load(f)
    sources = ["dtm","sar","lc"]
    conf['sources'] = sources
    model = LateFusion(conf)
    inputs = [torch.randn(1, conf["conf_"+source]['channels'][0], 256, 256) for source in sources]
    outputs = model(inputs)
