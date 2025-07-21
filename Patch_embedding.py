
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# Early Fusion
class DefaultPatchEmbed(nn.Module):
    
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, rgb_chans=3, hs_chans=4, dem_chans=1,  embed_dim=96, norm_layer=None):
        
        super().__init__()
        patch_size = to_2tuple(patch_size) # (4,4)
        self.patch_size = patch_size

        self.in_chans = rgb_chans + hs_chans + dem_chans # n_channel input: to change
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(self.in_chans, self.embed_dim, kernel_size=patch_size, stride=patch_size) # default=linear
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""

        rgb, hs, dem = x
        x = torch.cat([rgb, hs, dem],1)

        # padding
        _, _, H, W = x.size() # B, C, H, W
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww) # 16, 96, 64, 64

        return x

# Modality Token Patch Embdedding
class ModalityTokenPatchEmbed(nn.Module):
    
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, rgb_chans=3, hs_chans=4, dem_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size) # (4,4)
        self.patch_size = patch_size

        self.embed_dim = embed_dim

        self.rgb_chan = rgb_chans
        self.hs_chan = hs_chans
        self.dem_chan = dem_chans

        self.proj_rgb = nn.Conv2d(self.rgb_chan, int(self.embed_dim/3), kernel_size=patch_size, stride=patch_size) # default=linear
        self.proj_hs = nn.Conv2d(self.hs_chan, int(self.embed_dim/3), kernel_size=patch_size, stride=patch_size)
        self.proj_dem = nn.Conv2d(self.dem_chan, int(self.embed_dim/3), kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(int(self.embed_dim/3))
        else:
            self.norm = None

    def _padding(self, x, patch_size):

        _, _, H, W = x.size() # B, C, H, W
        if W % patch_size[1] != 0:
            x = F.pad(x, (0, patch_size[1] - W % patch_size[1]))
        if H % patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, patch_size[0] - H % patch_size[0]))
        
        return x    

    def forward(self, x):
        """Forward function."""

        rgb, hs, dem = x
        # rgb: 16, 3, 256, 256
        # hs: 16, 23, 256, 256
        # dem: 16, 1, 256, 256

        rgb = self._padding(rgb, self.patch_size)
        hs = self._padding(hs, self.patch_size)
        dem = self._padding(dem, self.patch_size)

        rgb = self.proj_rgb(rgb) # 16, 32, 64, 64
        hs = self.proj_hs(hs) # 16, 32, 64, 64
        dem = self.proj_dem(dem) # 16, 32, 64, 64

        if self.norm is not None:

            rgb = rgb.flatten(2).transpose(1, 2) # 16, 4096, 32
            rgb = self.norm(rgb)
            hs = hs.flatten(2).transpose(1, 2) # 16, 4096, 32
            hs = self.norm(hs)
            dem = dem.flatten(2).transpose(1, 2) # 16, 4096, 32
            dem = self.norm(dem)

            x = torch.cat((rgb, hs, dem), dim=2) # 16, 4096, 96 --> é giá normalizzato

            Wh, Ww = int(x.size(1)**(1/2)), int(x.size(1)**(1/2)) # 

            # x = self.norm(x)
            
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww) # 16, 96, 64, 64

        return x


# Normal Patch Embedding: No Fusion
class DefaultPatchEmbed_MF(nn.Module):
    
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=1,  embed_dim=96, norm_layer=None):

        super().__init__()
        patch_size = to_2tuple(patch_size) # (4,4)
        self.patch_size = patch_size

        self.in_chans = in_chans # n_channel input: to change
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(self.in_chans, self.embed_dim, kernel_size=patch_size, stride=patch_size) # default=linear
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""

        # padding
        _, _, H, W = x.size() # B, C, H, W
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww) # 16, 96, 64, 64

        return x


class ChannelPatchEmbed(nn.Module):

    def __init__(self, patch_size=4, rgb_chans=3, hs_chans=4, dem_chans=1, embed_dim=96, norm_layer=None):

        super().__init__()

        patch_size = to_2tuple(patch_size) # (4,4)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_chans = rgb_chans + hs_chans + dem_chans 

        self.proj = nn.Conv2d(1, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    

    def _padding(self, x, patch_size):

        _, _, H, W = x.size() # B, C, H, W
        if W % patch_size[1] != 0:
            x = F.pad(x, (0, patch_size[1] - W % patch_size[1]))
        if H % patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, patch_size[0] - H % patch_size[0]))
        
        return x   
    
    
    def forward(self, x):

        rgb, hs, dem = x
        x = torch.cat([rgb, hs, dem],1) # 16, 8, 256, 256

        outx = []
        
        for channel in range(0,self.in_chans): # controllare che passi su tutti i canali

            x_channel = x[:, channel, :, :] # select one channel --> B, 256, 256
            x_channel = torch.from_numpy(np.expand_dims(x_channel.cpu().detach().numpy(), axis=1)).cuda()
            x_channel = self._padding(x_channel, self.patch_size) # padding
            x_channel = self.proj(x_channel) # linear embedding --> B, 96, 64, 64
           
            if self.norm is not None:
                
                Wh, Ww = x_channel.size(2), x_channel.size(3) 
                x_channel = x_channel.flatten(2).transpose(1, 2) # B, 4096, 96
                x_channel = self.norm(x_channel) # B, 4096, 96
                x_channel = x_channel.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)  # B, 96, 64, 64
            
            outx.append(x_channel)
        
        x = torch.cat((outx[0], outx[1], outx[2], outx[3], outx[4], outx[5], outx[6], outx[7]), dim=1)
        
        return x
        