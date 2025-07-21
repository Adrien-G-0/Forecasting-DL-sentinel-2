import torch
import torch.nn as nn
import numpy as np
from transformers import SwinConfig, SwinModel ,UperNetConfig, UperNetForSemanticSegmentation
from class_module import *
from upernet import PSPModule, FPN_fuse
from itertools import chain

class SwinTranformers(nn.Module):
    def __init__(self, image_size = 256,
                        patch_size = 4,
                        num_channels = 3,
                        embed_dim = 60,
                        depths = [2, 2, 6, 2],
                        num_heads = [3, 6, 12, 24],
                        window_size = 7,
                        mlp_ratio = 4.0,
                        qkv_bias = True,
                        hidden_dropout_prob = 0.0,
                        attention_probs_dropout_prob = 0.0,
                        drop_path_rate = 0.1,
                        hidden_act = 'gelu',
                        use_absolute_embeddings = False,
                        initializer_range = 0.02,
                        layer_norm_eps = 1e-05,
                        encoder_stride = 32,
                        out_features = None,
                        out_indices = None ):
        super().__init__()

        configuration = SwinConfig(image_size = image_size,
                        patch_size = patch_size,
                        num_channels = num_channels,
                        embed_dim = embed_dim,
                        depths = depths,
                        num_heads = num_heads,
                        window_size = window_size,
                        mlp_ratio = mlp_ratio,
                        qkv_bias = qkv_bias,
                        hidden_dropout_prob = hidden_dropout_prob,
                        attention_probs_dropout_prob = attention_probs_dropout_prob,
                        drop_path_rate = drop_path_rate,
                        hidden_act = hidden_act,
                        use_absolute_embeddings = use_absolute_embeddings,
                        initializer_range = initializer_range,
                        layer_norm_eps = layer_norm_eps,
                        encoder_stride = encoder_stride,
                        out_features = out_features,
                        out_indices = out_indices )


        model = SwinModel(configuration,add_pooling_layer=False)

        self.encoder=model
        self.configuration = model.config

        self.bottle_neck=nn.Sequential(ConvBlock(480,480),ConvBlock(480,480))
        self.decodeur_1=UpsamplingBlock(480,240,480)
        self.decodeur_2=UpsamplingBlock(240,120,240)
        self.decodeur_3=UpsamplingBlock(120,60,120)
        self.decodeur_4=UpsamplingBlock(60,30,60)
        self.decodeur_5=UpsamplingBlock(30,1,0)
        

        

        

    def forward(self,x):
            x= torch.cat([input for input in x],dim=1)
            dict=self.encoder(x,output_hidden_states=True)
            reshaped_hidden_states = list(dict['reshaped_hidden_states'])
            x0 = reshaped_hidden_states.pop()
            x1=self.bottle_neck(x0)
            x2=self.decodeur_1(x1,reshaped_hidden_states.pop())
            x3=self.decodeur_2(x2,reshaped_hidden_states.pop())
            x4=self.decodeur_3(x3,reshaped_hidden_states.pop())
            x5=self.decodeur_4(x4,reshaped_hidden_states.pop())
            x6=self.decodeur_5(x5,torch.zeros(x5.size(0),0,x5.size(2),x5.size(3)))

            return x6
    

class Swin_UperNet(nn.Module):
    def __init__(self,
                 image_size=256,
                 patch_size=4,
                 num_channels=3,
                 embed_dim=60,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 hidden_dropout_prob=0.0,
                 attention_probs_dropout_prob=0.0,
                 drop_path_rate=0.1,
                 hidden_act='gelu',
                 use_absolute_embeddings=False,
                 initializer_range=0.02,
                 layer_norm_eps=1e-05,
                 encoder_stride=32,
                 out_features=None,
                 out_indices=None,
                 num_classes=1):  # Pour la régression, on utilise 1 sortie

        super().__init__()

        # Configuration Swin
        self.swin_config = SwinConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            drop_path_rate=drop_path_rate,
            hidden_act=hidden_act,
            use_absolute_embeddings=use_absolute_embeddings,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            encoder_stride=encoder_stride,
            out_features=out_features,
            out_indices=out_indices
        )
        # Encodeur Swin
        self.backbone = SwinModel(self.swin_config)
        feature_channels= [embed_dim * 2**i for i in range(len(depths))]  # Exemple de tailles de features
        self.PPN = PSPModule(in_channels=feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=feature_channels[0])
        self.head = nn.Conv2d(feature_channels[0], 1, kernel_size=3, padding=1)

    def forward(self, x):
        input_size = (256, 256)
        x=torch.cat([input for input in x], dim=1)  # Concatenate input channels if needed
        features = list(self.backbone(x,output_hidden_states=True)['reshaped_hidden_states'])
        # len(features) = 4
        # features[0].shape = torch.Size([16, , 64, 64])
        # features[1].shape = torch.Size([16, , 32, 32])
        # features[2].shape = torch.Size([16, , 16, 16])
        # features[3].shape = torch.Size([16, , 8, 8])
        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(features)) # 16, 9, 64, 64

        x = F.interpolate(x, size=input_size, mode='bilinear') # 16, 9, 256, 256
        return x

    #     # # Configuration UperNet
    #     # self.upernet_config = UperNetConfig(
    #     #     encoder_stride=encoder_stride,
    #     #     hidden_sizes=[256, 512, 1024, 2048],  # Exemple de tailles de features
    #     #     num_classes=num_classes  # Pour la régression, on utilise généralement 1
    #     # )

    #     # # Décodeur UperNet
    #     # self.upernet_decoder = UperNetForSemanticSegmentation(self.upernet_config)

    #     # # Remplacement de la tête de classification pour la régression
    #     # self.regression_head = nn.Sequential(
    #     #     nn.Conv2d(256, 128, kernel_size=3, padding=1),
    #     #     nn.ReLU(),
    #     #     nn.AdaptiveAvgPool2d(1),
    #     #     nn.Flatten(),
    #     #     nn.Linear(128, num_classes)  # Pour la régression, on utilise généralement 1 sortie
    #     # )


        

    # def forward(self, x):
    #     # Passage à travers l'encodeur Swin
    #     swin_outputs = self.swin_encoder(pixel_values=x,output_hidden_states=True)
    #     reshaped_hidden_states = list(swin_outputs['reshaped_hidden_states'])
    #     # Extraction des features à différentes échelles
    #     # Note: Ici, nous devrions extraire les features à différentes échelles de Swin
    #     # Pour l'instant, nous utilisons des tensors aléatoires comme placeholder
    #     # Dans une implémentation réelle, vous devrez extraire les vraies features
    #     features = list(swin_outputs.hidden_states[1:])

    #     # Passage à travers le décodeur UperNet
    #     upernet_output = self.upernet_decoder(pixel_values=features).logits

    #     # Passage à travers la tête de régression
    #     regression_output = self.regression_head(upernet_output)

    #     return regression_output
  

class UperNet_swin(nn.Module):

    # Implementing only the object path
    
    def __init__(self,image_size = 256,
                        patch_size = 4,
                        num_channels = 3,
                        embed_dim = 60,
                        depths = [2, 2, 6, 2],
                        num_heads = [3, 6, 12, 24],
                        window_size = 7,
                        mlp_ratio = 4.0,
                        qkv_bias = True,
                        hidden_dropout_prob = 0.0,
                        attention_probs_dropout_prob = 0.0,
                        drop_path_rate = 0.1,
                        hidden_act = 'gelu',
                        use_absolute_embeddings = False,
                        initializer_range = 0.02,
                        layer_norm_eps = 1e-05,
                        encoder_stride = 32,
                        out_features = None,
                        out_indices = None,
                        num_classes=1):  
        
        super().__init__()

        configuration = SwinConfig(image_size = image_size,
                        patch_size = patch_size,
                        num_channels = num_channels,
                        embed_dim = embed_dim,
                        depths = depths,
                        num_heads = num_heads,
                        window_size = window_size,
                        mlp_ratio = mlp_ratio,
                        qkv_bias = qkv_bias,
                        hidden_dropout_prob = hidden_dropout_prob,
                        attention_probs_dropout_prob = attention_probs_dropout_prob,
                        drop_path_rate = drop_path_rate,
                        hidden_act = hidden_act,
                        use_absolute_embeddings = use_absolute_embeddings,
                        initializer_range = initializer_range,
                        layer_norm_eps = layer_norm_eps,
                        encoder_stride = encoder_stride,
                        out_features = out_features,
                        out_indices = out_indices ) 



        self.backbone=SwinModel(configuration,add_pooling_layer=False)
        self.backbone_configuration = self.backbone.config
        
        feature_channels= [1]
        self.PPN = PSPModule(in_channels=feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=feature_channels[0])
        self.head = nn.Conv2d(feature_channels[0], num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        input_size = (256,256)

        features = self.backbone(x)
        # len(features) = 4
        # features[0].shape = torch.Size([16, 96, 64, 64])
        # features[1].shape = torch.Size([16, 192, 32, 32])
        # features[2].shape = torch.Size([16, 384, 16, 16])
        # features[3].shape = torch.Size([16, 768, 8, 8])
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





if __name__ == "__main__":
    num_channels=20
    x=torch.randn(1,num_channels,256,256)


    model=Swin_UperNet(num_channels=num_channels)
    y=model(x)
    print(sum(p.numel() for p in model.parameters()))
    # print(y.keys())
    # for hidden_sate,reshaped in zip(y['hidden_states'],y['reshaped_hidden_states']):
    #     print(hidden_sate.size(), 'reshaped',reshaped.size())
    print(y.size())


   


