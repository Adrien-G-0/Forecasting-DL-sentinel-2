import torch
import torch.nn as nn
import numpy as np
from torch.nn import Sequential
# from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel
# from transformers import  ViTConfig, ViTModel
from torchvision import transformsfroms
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
        


# class visionTransformer


# class TimeseriesTransformer