import torch
import torch.nn as nn
import numpy as np
import math
# from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel
# from transformers import  ViTConfig, ViTModel
# from torchvision import transforms
from torch.utils.data import DataLoader,TensorDataset  
import time


class PatchEmbedding(nn.Module):
    def __init__(self, img_size:int, in_channels:int, embed_dim:int,patch_size:int=16):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Projection linéaire des patches
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    def forward(self, x):
        
        x = self.projection(x)    # x shape: (batch_size, in_channels, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)    # x shape: (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)# x shape: (batch_size, n_patches, embed_dim)

        return x 
        
class TEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x
    


class VisionTransformer(nn.Module):
    def __init__(self, img_size:int, in_channels:int ,out_channels:int, embed_dim:int, num_heads:int=4, num_layers:int=4, patch_size:int=16, dropout:int=0.1):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.patch_embedding = PatchEmbedding(img_size, in_channels, embed_dim, patch_size)
        self.transformer_encoder = TEncoder(embed_dim, num_heads, num_layers, dropout)
        self.reg_token=nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embedding.n_patches + 1, embed_dim))

        self.decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, patch_size * patch_size * out_channels)  # Retourner au nombre de canaux original
        )

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embedding(x)

        reg_token = self.reg_token.expand(batch_size, -1, -1)
        x = torch.cat((reg_token, x), dim=1)
        x = x + self.pos_embed

        x = self.transformer_encoder(x)
        x = x[:, 1:,:]

        x = self.decoder(x)
        # Reformater en patches puis reconstruire l'image
        n_patches_sqrt = int(math.sqrt(self.patch_embedding.n_patches))
        x = x.view(batch_size, n_patches_sqrt, n_patches_sqrt, 
                   self.patch_size, self.patch_size, -1)
        
        # Permuter les dimensions pour reconstruire l'image
        x = x.permute(0, 5, 1, 3, 2, 4)  # (batch_size, channels, n_patches_sqrt, patch_size, n_patches_sqrt, patch_size)
        x = x.contiguous().view(batch_size, -1, self.img_size, self.img_size)
        # Appliquer la fonction d'activation
        x = self.sigmoid(x)
        return x


img_size=256
in_channels=4
out_channels=1
embed_dim=1024
num_heads=8
num_layers=4
patch_size=16
dropout=0.1


model = VisionTransformer(
    img_size=img_size,
    out_channels=out_channels,
    in_channels=in_channels,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    patch_size=patch_size,
    dropout=dropout
)
print(model)

input_tensor = torch.randn(1, in_channels, img_size, img_size)  # Example input
deb=time.time()
output = model(input_tensor)
time=time.time()-deb
print("Time taken for forward pass:", time)
print("Output shape:", output.shape)  # Should be (1, 1, patch_size, patch_size)














# class visionTransformer


# class TimeseriesTransformer