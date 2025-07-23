import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
import numpy as np



class TimeArchitecure(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(TimeArchitecure, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(in_channels, in_channels//2, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(in_channels//2, in_channels//2, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv1d(in_channels//2, in_channels, kernel_size=kernel_size, padding=kernel_size//2)

        self.time_matrix= nn.Parameter(torch.randn(1, in_channels//2, kernel_size), requires_grad=True)
        self.time_matrix.data.uniform_(-0.1, 0.1)
        self.time_matrix = self.time_matrix.to(torch.float32)

        self.linear = nn.Linear(in_channels, in_channels)

    def forward(self, x,time):
        # x shape: (batch_size, in_channels, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Transformation dépendant de `time`
        time_vector = self.time_transform(time.unsqueeze(1))  # Assure une forme correcte
        time_vector = time_vector.unsqueeze(2)  # Ajouter une dimension pour le broadcasting

        x = x @ time_vector  # Appliquer la modulation temporelle
        x = self.conv3(x)
        x = F.relu(x)

        x= self.linear(x)

        return x
    

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
            super().__init__()
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

        
class TimeBlock(nn.Module):
    """
    TimeBlock module that implements temporal and spatial transformation
    according to the formula: (t*A + B)*X + (t*C + D)
    where X is the input tensor, t is time, and A, B, C, D are parameter matrices
    """
    def __init__(self, in_channels,out_channels, spatial_size):
        super(TimeBlock, self).__init__()
        
        # Ensure that spatial_size is a list/tuple with 2 elements
        assert len(spatial_size) == 2, "spatial_size must be a list/tuple with 2 elements [h, w]"
        h, w = spatial_size
        
        # Parameter matrices for the linear time transformation
        # A and B are used for the multiplicative transformation (t*A + B)
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.activation = nn.Sigmoid()
        self.A = nn.Parameter(torch.randn(out_channels, h, h))
        self.B = nn.Parameter(torch.randn(out_channels, h, h))
        
        # C and D are used for the additive term (t*C + D)
        self.C = nn.Parameter(torch.randn(out_channels, h, w))
        self.D = nn.Parameter(torch.randn(out_channels, h, w))
        
        # Initialization for better convergence
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        nn.init.zeros_(self.C)
        nn.init.zeros_(self.D)
        
    
    def forward(self, x, t):
        """
        Applies the transformation (t*A + B)*X + (t*C + D)
        
        Args:
            x: Input tensor [batch_size, in_channels, h, w]
            t: Time tensor [batch_size] or scalar
        """
        # If t is a scalar, convert it to a tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x.device)
        
        # If t is a tensor of shape [batch_size], reshape it for broadcasting
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)  # [batch_size, 1, 1, 1]
        elif t.dim() == 0:  # If t is a scalar
            t = t.view(1, 1, 1, 1)
        
        # Compute the linear time transformation matrix: L = t*A + B
        # This matrix is computed for each channel and each batch element
        t = self.linear1(t)
        t = self.activation(t)
        L = torch.einsum('b...,chw->bchw', t, self.A) + self.B.unsqueeze(0)
        
        # Compute the additive term: P = t*C + D
        P = torch.einsum('b...,chw->bchw', t, self.C) + self.D.unsqueeze(0)
        
        # Apply the transformation: y = L*x + P
        # We use einsum to efficiently apply L to x
        # 'bchw,bcij->bcij' means we multiply L and x element-wise over dimensions c, h, w
        transformed_x = torch.einsum('bchw,bcij->bcij', L, x) + P
        
        return transformed_x
        




#TODO# Add the time transformation to the bottleneck
class TimeArchitecure(torch.nn.Module):
    def __init__(self,input_channels):
        super().__init__()
        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)


        self.bottleneck = Sequential(ConvBlock(512, 1024), ConvBlock(1024, 512))
        self.time_transformation= TimeBlock(1,512,(16,16))
        self.decoder1 = UpsamplingBlock(512, 256, 512)
        self.decoder2 = UpsamplingBlock(256, 128, 256)
        self.decoder3 = UpsamplingBlock(128, 64, 128)
        self.decoder4 = UpsamplingBlock(64, 1, 64)

        self.activation = torch.nn.Sigmoid()

    def forward(self, x, time):
        x, residual1 = self.encoder1(x)
        x, residual2 = self.encoder2(x)
        x, residual3 = self.encoder3(x)
        x, residual4 = self.encoder4(x)

        x = self.bottleneck(x)
        x = self.time_transformation(x,time)
        

        x = self.decoder1(x,residual4)
        x = self.decoder2(x,residual3)
        x = self.decoder3(x,residual2)
        output = self.decoder4(x,residual1)


        output=self.activation(output)  # Test to remoove the activation function because the output is alsmost close to 0 and 1
        return 2*output-1 # to return NDVI value between -1 and 1
    



if __name__ == "__main__":
    
    #test
    test=TimeArchitecure(3)
    time=2.
    tensor=torch.randn(1, 3, 256, 256)
    pol=nn.MaxPool2d(kernel_size=2,stride=2)

    output= test(tensor,time)

    # tensor=torch.randn(1, 64, 256, 256)
    # timeArchitecure=TimeArchitecure(64)
    # output=timeArchitecure(tensor,time)
    print(output.shape)  # Should be (1, 64, 256, 256)