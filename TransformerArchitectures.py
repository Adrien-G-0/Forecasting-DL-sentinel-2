import torch
import torch.nn as nn
import numpy as np
# from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel
# from transformers import  ViTConfig, ViTModel
# from torchvision import transforms
from torch.utils.data import DataLoader,TensorDataset  



class PatchEmbedding(nn.Module):
    def __init__(self, img_size, in_channels, embed_dim,patch_size=16):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        
        x = self.proj(x)    # x shape: (batch_size, in_channels, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)    # x shape: (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)# x shape: (batch_size, n_patches, embed_dim)

        return x 
        
class TEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=embed_dim, 
            dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x
    


class VisionTransformer(nn.Module):
    def __init__(self, img_size, in_channels , embed_dim, num_heads=4, num_layers=4, patch_size=16, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, in_channels, embed_dim, patch_size)
        self.transformer_encoder = TEncoder(embed_dim, num_heads, num_layers, dropout)
        self.reg_token=nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embedding.n_patches + 1, embed_dim))

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),

            nn.Linear(embed_dim, patch_size*patch_size*1),
            nn.Unflatten(2, (1,patch_size, patch_size)),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embedding(x)

        reg_token = self.reg_token.expand(batch_size, -1, -1)
        x = torch.cat((reg_token, x), dim=1)
        x = x + self.pos_embed

        x = self.transformer_encoder(x)
        x = x[:, 1:,:]
        x = self.decoder(x)
        return x


model = VisionTransformer(
    img_size=64,
    in_channels=3,
    embed_dim=512,
    num_heads=4,
    num_layers=4,
    patch_size=16,
    dropout=0.1
)
print(model)
# class visionTransformer


# class TimeseriesTransformer