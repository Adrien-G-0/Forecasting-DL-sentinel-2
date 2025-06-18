import torch
import torch.nn as nn
import numpy as np
from transformers import SwinConfig, SwinModel ,UperNetConfig, UperNetForSemanticSegmentation
from class_module import *


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
        self.swin_encoder = SwinModel(self.swin_config, add_pooling_layer=False)

        # Configuration UperNet
        self.upernet_config = UperNetConfig(
            encoder_stride=encoder_stride,
            hidden_sizes=[256, 512, 1024, 2048],  # Exemple de tailles de features
            num_classes=num_classes  # Pour la régression, on utilise généralement 1
        )

        # Décodeur UperNet
        self.upernet_decoder = UperNetForSemanticSegmentation(self.upernet_config)

        # Remplacement de la tête de classification pour la régression
        self.regression_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)  # Pour la régression, on utilise généralement 1 sortie
        )

    def forward(self, x):
        # Passage à travers l'encodeur Swin
        swin_outputs = self.swin_encoder(pixel_values=x)

        # Extraction des features à différentes échelles
        # Note: Ici, nous devrions extraire les features à différentes échelles de Swin
        # Pour l'instant, nous utilisons des tensors aléatoires comme placeholder
        # Dans une implémentation réelle, vous devrez extraire les vraies features
        features = [
            torch.randn(1, 256, x.size(2)//4, x.size(3)//4),  # Exemple de feature à échelle 1/4
            torch.randn(1, 512, x.size(2)//8, x.size(3)//8),  # Exemple de feature à échelle 1/8
            torch.randn(1, 1024, x.size(2)//16, x.size(3)//16),  # Exemple de feature à échelle 1/16
            torch.randn(1, 2048, x.size(2)//32, x.size(3)//32)  # Exemple de feature à échelle 1/32
        ]

        # Passage à travers le décodeur UperNet
        upernet_output = self.upernet_decoder(features=features).logits

        # Passage à travers la tête de régression
        regression_output = self.regression_head(upernet_output)

        return regression_output



if __name__ == "__main__":
    x=torch.randn(1,2,256,256)
    x1=torch.randn(1,1,256,256)
    config=SwinConfig(image_size=256)
    model=SwinModel(config)
    model=SwinTranformers()
    y=model([x,x1])#,output_hidden_states=True)
    print(sum(p.numel() for p in model.parameters()))
    # print(y.keys())
    # for hidden_sate,reshaped in zip(y['hidden_states'],y['reshaped_hidden_states']):
    #     print(hidden_sate.size(), 'reshaped',reshaped.size())
    print(y.size())


    model = Swin_UperNet()

    # Créer un tensor d'entrée d'exemple (batch_size=1, channels=3, height=256, width=256)
    input_tensor = torch.randn(1, 3, 256, 256)

    # Passer à travers le modèle
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Devrait être [1, 1] pour une régression simple


