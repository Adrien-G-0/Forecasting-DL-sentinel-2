# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from itertools import chain

# from swin import SwinTransformer
# from swin_EF import SwinTransformer
from swin_CF import SwinTransformer
from upernet import PSPModule, FPN_fuse

import json
with open('../Fusion-Methods-RS/config.json', 'r') as f:
  config = json.load(f)

class UperNet_swin(nn.Module):

    # Implementing only the object path
    
    def __init__(self,
                 params:dict, 
                 num_classes=9): # c'erano altri parametri !
        
        super(UperNet_swin, self).__init__()

        self.config = params 

        self.backbone = SwinTransformer(pretrain_img_size= self.config["train_size"][0], 
                                        patch_size=self.config["patch_size"],
                                        rgb_chans=self.config["rgb_chans"], 
                                        hs_chans=self.config["hs_chans"],
                                        dem_chans=self.config["dem_chans"],
                                        embed_dim = self.config["embed_dim_CF"],
                                        embed_dim_after_CF = self.config["embed_dim_after_CF"],
                                        depths = self.config["depth_S_B_L"],
                                        num_heads = self.config["num_heads"],
                                        window_size = self.config["window_size"],
                                        mlp_ratio = self.config["mlp_ratio"],
                                        qkv_bias=True,
                                        qk_scale=None,
                                        drop_rate= self.config["drop_rate"],
                                        attn_drop_rate= self.config["attn_drop_rate"],
                                        drop_path_rate= self.config["drop_path_rate"],
                                        norm_layer=nn.LayerNorm,
                                        ape=False,
                                        patch_norm=True,
                                        out_indices=(0, 1, 2, 3),
                                        # frozen_stages=-1,
                                        use_checkpoint=False)
        
        feature_channels= self.config["features_channels_upernet_CF"]
        self.PPN = PSPModule(in_channels=feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=feature_channels[0])
        self.head = nn.Conv2d(feature_channels[0], num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        input_size = (self.config["train_size"][0], self.config["train_size"][1])

        features = self.backbone(x)
        # len(features) = 4
        # features[0].shape = torch.Size([16, 192, 64, 64])
        # features[1].shape = torch.Si  ze([16, 384, 32, 32])
        # features[2].shape = torch.Size([16, 768, 16, 16])
        # features[3].shape = torch.Size([16, 2*768, 8, 8])
        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(features)) # 16, 9, 64, 64

        x = F.interpolate(x, size=input_size, mode='bilinear') # 16, 9, 256, 256
        return x

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()



if __name__ == '__main__':
    model = UperNet_swin(params=config, num_classes=1)
    model = model.cuda()
    rgb = torch.randn(8, 3, 256, 256,device='cuda')
    hs= torch.randn(8, 4, 256, 256,device='cuda')
    dem = torch.randn(8, 1, 256, 256,device='cuda')
    x = (rgb, hs, dem)  # Input tuple of tensors
    y = model(x)
    print(y.shape)  # Expected output: torch.Size([16, 9, 256, 256])