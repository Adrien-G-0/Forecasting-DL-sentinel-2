import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import math
import time
import pickle
import numpy as np

from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A

from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel
from transformers import ViTConfig, ViTModel
from NewBase import Base
from middle_fusion_ import Middle_fusion_en as mf_
import pytorch_lightning as pl

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar


class TransformerArchitectures(Base):
    def __init__(self, params):
        # init base
        super(TransformerArchitectures, self).__init__(params)
        
        # reorganize sources values
        source_order = ['rgb', 'hs', 'dtm', 'sar','lc','sau']
        ordered_sources = []
        for source in source_order:
            if source in self.conf['sources']:
                ordered_sources.append(source)
        self.conf['sources'] = ordered_sources

        # early fusion
        if self.conf['method'] == 'early_fusion':

            input_channels = 0
            for source in self.conf['sources']:
                if source == 'rgb':
                    input_channels = input_channels + 3
                if source == 'hs':
                    input_channels = input_channels + 182
                if source =='dtm':
                    input_channels = input_channels + 1
                if source == 'sar' :
                    input_channels = input_channels +2
                if source == 'lc':
                    input_channels = input_channels +self.conf['num_class_lc']  # should be 8
                if source == 'sau':
                    input_channels = input_channels +self.conf['num_class_sau']  # should be 10
        
            # define architecture
            self.net=ViTWithDecoder(input_size=self.conf['train_size'][0],
                                    num_patches=self.conf['num_patches'],
                                    embed_dim=self.conf['embed_dim'],
                                    embedding_type=self.conf['embedding_type'],
                                    dropout=self.conf['dropout'],
                                    in_channels=input_channels,
                                    patch_size=self.conf['patch_size'],
                                    num_layers_transformers=self.conf['num_layers_transformers'],
                                    num_heads=self.conf['num_heads'],
                                    mlp_ratio=self.conf['mlp_ratio'],
                                    attention_dropout=self.conf['attention_dropout'],
                                    drop_path_rate=self.conf['drop_path_rate'],
                                    norm_layer=nn.LayerNorm,
                                    activation=self.conf['activation'],
                                    return_attention=False
            )
            # Initialization_weight TODO


        # middle fusion
        elif self.conf['method'] == 'middle_fusion':
            # TODO change the dimensio embedind to 8 or 16
            sources = self.conf['sources']
            self.fusion_en = mf_(self.conf)
            in_channels_middle_fusion = np.sum(self.conf['conf_'+source]["channels"][-1] for source in sources)  # last channel of each source 
            
            # define architecture
            self.net = ViTWithDecoder(input_size=self.conf['train_size'][0],
                                    num_patches=self.conf['num_patches'],
                                    embed_dim=self.conf['embed_dim'],
                                    embedding_type=self.conf['embedding_type'],
                                    dropout=self.conf['dropout'],
                                    in_channels=in_channels_middle_fusion,
                                    patch_size=self.conf['patch_size'],
                                    num_layers=self.conf['num_layers'],
                                    num_heads=self.conf['num_heads'],
                                    mlp_ratio=self.conf['mlp_ratio'],
                                    attention_dropout=self.conf['attention_dropout'],
                                    drop_path_rate=self.conf['drop_path_rate'],
                                    norm_layer=nn.LayerNorm,
                                    activation=self.conf['activation'],
                                    return_attention=False)
            # Initialization_weight TODO


        self.mean_dict = self.load_dict(self.conf['mean_dict_01'])
        self.std_dict = self.load_dict(self.conf['std_dict_01'])
        self.max_dict = self.load_dict(self.conf['max_dict_01'])
        self.loaded_min_dict_before_normalization = self.load_dict(self.conf['min_dict'])
        self.loaded_max_dict_before_normalization = self.load_dict(self.conf['max_dict'])

    def load_dict(self, name):
        with open(name, 'rb') as f:
            loaded_dict = pickle.load(f)

        return loaded_dict
    
    def forward(self, batch): 

        components = {}
        for source, input_tensor in zip(self.conf['sources'], batch):
            components[source] = input_tensor
            
        
        if self.conf['method'] == 'early_fusion':
            first_flag = True
            inp = None
        
            for source in self.conf['sources']:
                if first_flag:
                    inp = components[source]
                    first_flag = False
                else:
                    inp = torch.cat([inp, components[source]], axis=1)

            with torch.device("meta"):
                model = self.net
                x = inp

            model_fwd = lambda: model(x)

            output = self.net(inp)
            return output
        
        
        elif self.conf['method'] == 'middle_fusion':
            

            inp = self.fusion_en(batch)

            with torch.device("meta"):
                model = self.net
                x = inp

            model_fwd = lambda: model(x)

            output = self.net(inp)
            
            return output     



    def create_transform_function(self, transform_list):
        # create transformation function
        def transform_inputs(inps):
            # create transformation
            sources_possibles = ['rgb', 'hs', 'dtm', 'sar','lc','sau', 'ndvi']
            inps_dict = {source: inps[i] for i, source in enumerate(self.conf['sources']+['ndvi'])}  # add ndvi to the inputs dict

            # Checking if all keys have a designated value else 0 TODO can maybe be improve to reduce storage and calculations
            inps_dict = {source: inps_dict.get(source, torch.zeros((1,))) for source in sources_possibles}
            rgb, hs, dtm, sar, lc, sau, ndvi = inps_dict['rgb'], inps_dict['hs'], inps_dict['dtm'], inps_dict['sar'], inps_dict['lc'],inps_dict['sau'], inps_dict['ndvi']


            normalize_rgb, normalize_hs, normalize_dtm, normalize_sar, transforms_augmentation = transform_list
            #no normalization for ndvi because it is already between -1 and 1
            ndvi=ndvi.unsqueeze(2) # so ndvi has the same shape as the others
            # no normalization for lc and sau because it is onehot encoded


            rgb = (rgb.numpy() - self.loaded_min_dict_before_normalization['rgb']) / (self.loaded_max_dict_before_normalization['rgb'] - self.loaded_min_dict_before_normalization['rgb'])
            hs = (hs.numpy() - self.loaded_min_dict_before_normalization['hs']) / (self.loaded_max_dict_before_normalization['hs'] - self.loaded_min_dict_before_normalization['hs'])
            dtm = (dtm.numpy() - self.loaded_min_dict_before_normalization['dtm']) / (self.loaded_max_dict_before_normalization['dtm'] - self.loaded_min_dict_before_normalization['dtm'])
            sar = (sar.numpy() - self.loaded_min_dict_before_normalization['sar']) / (self.loaded_max_dict_before_normalization['sar'] - self.loaded_min_dict_before_normalization['sar'])
            #no need to normalize the ndvi because it is already between -1 and 1 and lc,sau are onehot encoded
            ndvi = ndvi.numpy()
            lc=lc.numpy()
            sau=sau.numpy()

            rgb = rgb.astype(np.float32)
            hs = hs.astype(np.float32)
            dtm = dtm.astype(np.float32)
            sar = sar.astype(np.float32)
            ndvi = ndvi.astype(np.float32)
            lc = lc.astype(np.float32)
            sau = sau.astype(np.float32)
            
            rgb = normalize_rgb(image=rgb)['image']
            hs = normalize_hs(image=hs)['image']
            dtm = normalize_dtm(image=dtm)['image']
            sar = normalize_sar(image=sar)['image']


            # initialize the transforms
            transforms = A.Compose([transforms_augmentation], is_check_shapes=False,
                                    additional_targets={'hs': 'image',
                                                        'dtm': 'image',
                                                        'sar': 'image',
                                                        'lc': 'image',
                                                        'sau': 'image',
                                                        'ndvi': 'image'})
            # apply the transforms
            sample = transforms(image=rgb,
                                hs=hs,
                                dtm=dtm,
                                sar=sar,
                                lc=lc,
                                sau=sau,
                                ndvi=ndvi
                                
                                )
            # get images
            rgb = sample['image']
            hs = sample['hs']
            dtm = sample['dtm']
            sar = sample['sar']
            lc= sample['lc']
            sau = sample['sau']
            ndvi = sample['ndvi']

            outputs_dict = {'rgb': rgb, 'hs': hs, 'dtm': dtm, 'sar': sar ,'lc': lc, 'sau':sau, 'ndvi': ndvi}
            # get needed output values without the others
            output = list(outputs_dict[source] for source in self.conf['sources']) + [ndvi]
            return output
            

        # return the function
        return transform_inputs

    def train_transforms(self):
        # define training size
        train_size = self.conf['train_size'] if 'train_size' in self.conf else self.conf['input_size']
        # create transformation

        normalize_rgb = A.Normalize(mean=self.mean_dict['rgb'], std=self.std_dict['rgb'], max_pixel_value=self.max_dict['rgb'])
        normalize_hs = A.Normalize(mean=self.mean_dict['hs'], std=self.std_dict['hs'], max_pixel_value=self.max_dict['hs'])
        normalize_dtm = A.Normalize(mean=self.mean_dict['dtm'], std=self.std_dict['dtm'], max_pixel_value=self.max_dict['dtm'])
        normalize_sar = A.Normalize(mean=self.mean_dict['sar'], std=self.std_dict['sar'], max_pixel_value=self.max_dict['sar'])

        transforms_augmentation = A.Compose([A.Resize(*self.conf['input_size']),
            A.crops.transforms.RandomCrop(*train_size),
            A.Rotate(limit=[-180, 180]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            ToTensorV2()
        ], is_check_shapes=False)

        transforms = normalize_rgb, normalize_hs, normalize_dtm, normalize_sar, transforms_augmentation

        # create transform function
        return self.create_transform_function(transforms)
        

    def val_transforms(self):
        
        # create transformation
        normalize_rgb = A.Normalize(mean=self.mean_dict['rgb'], std=self.std_dict['rgb'], max_pixel_value=self.max_dict['rgb'])
        normalize_hs = A.Normalize(mean=self.mean_dict['hs'], std=self.std_dict['hs'], max_pixel_value=self.max_dict['hs'])
        normalize_dtm = A.Normalize(mean=self.mean_dict['dtm'], std=self.std_dict['dtm'], max_pixel_value=self.max_dict['dtm'])
        normalize_sar = A.Normalize(mean=self.mean_dict['sar'], std=self.std_dict['sar'], max_pixel_value=self.max_dict['sar'])

        transforms_augmentation = A.Compose([
            A.Resize(*self.conf['input_size']),
            ToTensorV2()
        ], is_check_shapes=False)

        transforms = normalize_rgb, normalize_hs, normalize_dtm, normalize_sar, transforms_augmentation
        
        # create transform function
        return self.create_transform_function(transforms)
    
    def test_transforms(self):
        return self.val_transforms()




class PositionalEmbedding(nn.Module):
    """
    Positional Embedding pour Vision Transformer (ViT)
    Supporte les embeddings apprenables et sinusoïdaux
    """
    
    def __init__(self, 
                 num_patches, 
                 embed_dim, 
                 embedding_type='learnable',
                 dropout=0.1):
        """
        Args:
            num_patches (int): Nombre de patches dans l'image
            embed_dim (int): Dimension des embeddings
            embedding_type (str): 'learnable' ou 'sinusoidal'
            dropout (float): Taux de dropout
        """
        super().__init__()
        
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.embedding_type = embedding_type
        
        
        # Nombre total de positions (patches )
        self.num_positions = num_patches 
        
        if embedding_type == 'learnable':
            # Embeddings apprenables (approche standard ViT)
            self.pos_embedding = nn.Parameter(
                torch.randn(1, self.num_positions, embed_dim) * 0.02
            )
        elif embedding_type == 'sinusoidal':
            # Embeddings sinusoïdaux fixes (comme dans les Transformers originaux)
            self.register_buffer('pos_embedding', 
                               self._create_sinusoidal_embeddings())
        else:
            raise ValueError("embedding_type doit être 'learnable' ou 'sinusoidal'")
            
        self.dropout = nn.Dropout(dropout)
    
    def _create_sinusoidal_embeddings(self):
        """Crée les embeddings positionnels sinusoïdaux"""
        pe = torch.zeros(self.num_positions, self.embed_dim)
        position = torch.arange(0, self.num_positions).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() *
                           -(math.log(10000.0) / self.embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Ajoute la dimension batch
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensor d'entrée de shape (batch_size, seq_len, embed_dim)
                             où seq_len = num_patches
        
        Returns:
            torch.Tensor: x + positional embeddings avec dropout appliqué
        """
        batch_size, seq_len, _ = x.shape
        
        # Vérification des dimensions
        if seq_len != self.num_positions:
            raise ValueError(f"Longueur de séquence {seq_len} ne correspond pas "
                           f"au nombre de positions {self.num_positions}")
        
        # Ajoute les embeddings positionnels
        # pos_embedding = self.pos_embedding.expand(batch_size, seq_len, self.embed_dim)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)
    
    def interpolate_pos_embedding(self, new_num_patches):
        """
        Interpole les embeddings positionnels pour une nouvelle taille d'image
        Utile pour le fine-tuning avec des résolutions différentes
        """
        if self.embedding_type != 'learnable':
            raise NotImplementedError("L'interpolation n'est supportée que pour les embeddings apprenables")
        
        old_num_patches = self.num_patches
        
        if old_num_patches == new_num_patches:
            return
        
        
        patch_pos_embed = self.pos_embedding
        
        # Calcule les dimensions de la grille
        old_grid_size = int(math.sqrt(old_num_patches))
        new_grid_size = int(math.sqrt(new_num_patches))
        
        # Reshape pour interpolation 2D
        patch_pos_embed = patch_pos_embed.reshape(
            1, old_grid_size, old_grid_size, self.embed_dim
        ).permute(0, 3, 1, 2)  # (1, embed_dim, H, W)
        
        # Interpolation bilinéaire
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed,
            size=(new_grid_size, new_grid_size),
            mode='bilinear',
            align_corners=False
        )
        
        # Reshape vers la forme originale
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(
            1, new_num_patches, self.embed_dim
        )
        
        # Reconstruit les embeddings complets
        new_pos_embedding = patch_pos_embed
        
        # Met à jour les paramètres
        self.pos_embedding = nn.Parameter(new_pos_embedding)
        self.num_patches = new_num_patches
        self.num_positions = new_num_patches + (1 if self.include_cls_token else 0)


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
        
class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class MultiHeadSelfAttention(nn.Module):
    """
    Module d'auto-attention multi-têtes avec nombre de têtes variable
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) doit être divisible par num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Projections pour Q, K, V
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        
        # Projection de sortie
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            mask: [batch_size, seq_len, seq_len] ou None
        Returns:
            output: [batch_size, seq_len, embed_dim]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        B, N, C = x.shape
        
        # Calcul de Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        
        # Calcul des scores d'attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        
        # Application du masque si fourni
        if mask is not None:
            if mask.dim() == 3:  # [B, N, N]
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Application de l'attention aux valeurs
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        
        # Projection finale
        output = self.proj(attn_output)
        output = self.proj_dropout(output)
        
        return output, attn_weights

class FeedForward(nn.Module):
    """
    Feed-Forward Network avec activation GELU
    """
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.1, activation='gelu'):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Activation {activation} none supported, use 'gelu', 'relu' or 'swish'")
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Bloc Transformer avec auto-attention et feed-forward
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, 
                 drop_path=0.0, norm_layer=nn.LayerNorm, activation='gelu'):
        super().__init__()
        
        self.norm1 = norm_layer(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = norm_layer(embed_dim)
        self.mlp = FeedForward(embed_dim, mlp_ratio, dropout, activation)
        
        # DropPath (Stochastic Depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x, mask=None):
        # Auto-attention avec connexion résiduelle
        attn_output, attn_weights = self.attn(self.norm1(x), mask)
        x = x + self.drop_path(attn_output)
        
        # Feed-forward avec connexion résiduelle
        mlp_output = self.mlp(self.norm2(x))
        x = x + self.drop_path(mlp_output)
        
        return x, attn_weights

class TransformerEncoder(nn.Module):
    """
    Encodeur Transformer avec auto-attention et paramètres flexibles
    """
    def __init__(self, 
                 embed_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm,
                 activation='gelu',
                 return_attention=False):
        """
        Args:
            embed_dim (int): Dimension des embeddings
            num_layers (int): Nombre de couches Transformer
            num_heads (int): Nombre de têtes d'attention
            mlp_ratio (float): Ratio pour la dimension cachée du MLP
            dropout (float): Taux de dropout
            drop_path_rate (float): Taux de drop path (stochastic depth)
            norm_layer: Couche de normalisation
            activation (str): Fonction d'activation ('gelu', 'relu', 'swish')
            return_attention (bool): Si True, retourne les poids d'attention
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.return_attention = return_attention
        
        # Drop path rates (augmente linéairement avec la profondeur)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        
        # Blocs Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                activation=activation
            ) for i in range(num_layers)
        ])
        
        # Normalisation finale
        self.norm = norm_layer(embed_dim)
        
        # Initialisation des poids
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            mask: [batch_size, seq_len, seq_len] ou None
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
            attention_weights: Liste des poids d'attention si return_attention=True
        """
        attention_weights = []
        list_residuals = []
        for block in self.blocks:
            x, attn_weights = block(x, mask)
            list_residuals.append(x) #saving residuals for skip_connections
            if self.return_attention:
                attention_weights.append(attn_weights)
        
        x = self.norm(x)
        
        if self.return_attention:
            return x, attention_weights
        else:
            return x , list_residuals
    
    def get_attention_maps(self, x, layer_idx=None):
        """
        Récupère les cartes d'attention pour une couche spécifique
        """
        attention_weights = []
        
        for i, block in enumerate(self.blocks):
            x, attn_weights = block(x)
            attention_weights.append(attn_weights)
            
            if layer_idx is not None and i == layer_idx:
                break
        
        if layer_idx is not None:
            return x, attention_weights[layer_idx]
        else:
            return x, attention_weights

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, hidden_dim, drop_path_rate=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, drop_path_rate)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

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

class UpsamplingBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(UpsamplingBlock, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_block1 = ConvBlock(in_channels, in_channels)
            self.conv_block2 = ConvBlock(in_channels, out_channels)

        def forward(self, x):
            x = self.upsample(x)
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            return x


class CNNDecoder(nn.Module):
    def __init__(self, patch_size,output_size,embed_dim, num_patches):
        super().__init__()
        self.patch_size = patch_size
        self.output_size = output_size 
        self.num_patches = num_patches
        self.num_layers = int(np.log2(output_size // patch_size))  # Nombre de couches d'upsampling
        self.dimension_reduction_rate = int(np.exp( np.log(embed_dim) / self.num_layers)) # Réduction de dimension par couche
        self.upsampling_blocks = nn.ModuleList()
        
        in_channels = embed_dim
        for i in range(self.num_layers-1):
            out_channels = in_channels // self.dimension_reduction_rate
            self.upsampling_blocks.append(
                UpsamplingBlock(in_channels, out_channels)
            )
            in_channels = out_channels
        self.upsampling_blocks.append(UpsamplingBlock(in_channels, 1))

        self.activation = nn.Sigmoid()  # Activation pour la sortie finale

    def forward(self, x,skip_residuals):
        """
        Args:
            x: [batch_size, num_patches, embed_dim]
        
        Returns:
            output: [batch_size, output_size, output_size]
        """
        batch_size = x.size(0)
        
        # Reshape en patches
        n_patches_sqrt = int(math.sqrt(self.num_patches))
        x = x.view(batch_size, n_patches_sqrt, n_patches_sqrt, -1)
        
        # Permuter les dimensions pour CNN
        x = x.permute(0, 3, 1, 2)
        # Passer à travers les blocs d'upsampling

        for i,block in enumerate(self.upsampling_blocks):
            if i+len(skip_residuals) > self.num_layers:
                # add skip_connection
                x = x + skip_residuals[i+len(skip_residuals)-self.num_layers]
            x = block(x)


        x= 2*self.activation(x) -1 # Appliquer l'activation finale
        # Retourner la sortie finale
        return x


class ReversePatchEmbedding(nn.Module):
    def __init__(self, out_img_size: int, out_channels: int, embed_dim: int, patch_size: int=16):
        super().__init__()
        self.out_img_size = out_img_size
        self.patch_size = patch_size
        self.n_patches = (out_img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Convolution transposée pour reconstruire l'image
        self.reverse_projection = nn.ConvTranspose2d(
            embed_dim, out_channels,
            kernel_size=out_img_size//patch_size,
            stride=out_img_size//patch_size,
            dilation=1
            
        )
       
    def forward(self, x):
        # x shape: (batch_size, num_patches, embed_dim)
        
        batch_size, num_patches,embed_dim = x.shape

        # Réorganiser les patches en une grille 2D
        x = x.view(batch_size, int(num_patches**0.5), int(num_patches**0.5), embed_dim)
        x = x.permute(0, 3, 1, 2)  # x shape: (batch_size, embed_dim, out_img_size//patch_size, out_img_size//patch_size)

        # Appliquer la convolution transposée pour obtenir l'image finale
        x = self.reverse_projection(x)  # x shape: (batch_size, out_channels, out_img_size, out_img_size)

        return x


class Skip_Connection(nn.Module):
    def __init__(self, num_skip_connection, embed_dim, list_in_channels,patch_size=16, out_img_size=128):
        super().__init__()
        assert len(list_in_channels) == num_skip_connection, "list_in_channels must match num_layers"
        self.num_layers = num_skip_connection
        self.list_in_channels = list_in_channels
        self.reverse_embedding = nn.ModuleList([ ReversePatchEmbedding(out_img_size= int(out_img_size* 2**(-num_skip_connection+k+1)),
                                                                   out_channels=list_in_channels[-k-1],
                                                                   embed_dim=embed_dim,
                                                                   patch_size=patch_size
                                                        )   for k in range(num_skip_connection)])

    def forward(self, list_residuals):
        """
        Args:
            x: [batch_size, patch_size, patch_size, embed_dim] ordered by deepth of creation, the first element comes from the first layer and therefore use last in the decoder

        Returns:
            list_outputs: list of the residuals ordered by deepth of use, the first element comes from the last layer and therefore use first in the decoder
        
        """
        assert len(list_residuals) == self.num_layers, "list_residuals must match num_layers"
        # The first residual is used for the last layer of the decoder, so it is used through the last skipped connection
        list_outputs = []
        for res,skip_connection in zip(list_residuals, self.reverse_embedding):
            x = skip_connection(res)
            list_outputs.append(x)
        return list_outputs

# #New test
# class ViTWithDecoder2(nn.Module):
#     def __init__(self, 
#                  input_size,
#                  num_patches, 
#                  embed_dim, 
#                  embedding_type='learnable',
#                  dropout=0.1,
#                  in_channels=2, 
#                  patch_size:int=16,                 
#                  num_layers=4,
#                  num_heads=8,
#                  mlp_ratio=4.0,
#                  attention_dropout=0.1,
#                  drop_path_rate=0.0,
#                  norm_layer=nn.LayerNorm,
#                  activation='gelu',
#                  return_attention=False):
#         super().__init__()
#         self.patch_embedding_class = PatchEmbedding(img_size=input_size,
#                                               in_channels=18,
#                                               embed_dim=embed_dim//2,
#                                               patch_size=patch_size)
        
#         self.patch_embedding_continuous = PatchEmbedding(img_size=input_size,
#                                               in_channels=2,
#                                               embed_dim=embed_dim-embed_dim//2,
#                                               patch_size=patch_size)
        
#         self.positional_embedding = PositionalEmbedding(num_patches=num_patches,
#                                                         embed_dim=embed_dim,
#                                                         embedding_type=embedding_type,
#                                                         dropout=dropout)
        
#         self.Tclass=TransformerEncoder(embed_dim=embed_dim//2,
#                                           num_layers=2,
#                                           num_heads=num_heads,
#                                           mlp_ratio=mlp_ratio,
#                                           dropout=attention_dropout,
#                                           drop_path_rate=drop_path_rate,
#                                           norm_layer=norm_layer,
#                                           activation=activation,
#                                           return_attention=return_attention)
        
#         self.Tcontinuous=TransformerEncoder(embed_dim=embed_dim-embed_dim//2,
#                                             num_layers=2,
#                                             num_heads=num_heads,
#                                             mlp_ratio=mlp_ratio,
#                                             dropout=attention_dropout,
#                                             drop_path_rate=drop_path_rate,
#                                             norm_layer=norm_layer,
#                                             activation=activation,
#                                             return_attention=return_attention)

#         self.encoder = TransformerEncoder(embed_dim=embed_dim,
#                                           num_layers=num_layers-2,
#                                           num_heads=num_heads,
#                                           mlp_ratio=mlp_ratio,
#                                           dropout=attention_dropout,
#                                           drop_path_rate=drop_path_rate,
#                                           norm_layer=norm_layer,
#                                           activation=activation,
#                                           return_attention=return_attention)
        

#         self.decoder = CNNDecoder(patch_size=patch_size,output_size=input_size,embed_dim=embed_dim, num_patches=num_patches)
        
#         # self.pseudo_skip_connection = ConvBlock(in_channels=in_channels, out_channels=1)

#     def forward(self, x):
#         x_class = x[:,2:]
#         x_continuous = x[:,:2]
#         x_class_emb = self.patch_embedding_class(x_class)
#         x_continuous_emb = self.patch_embedding_continuous(x_continuous)

#         x_emb= torch.cat([x_continuous_emb,x_class_emb], dim=-1)  # Concaténer les embeddings
#         x_emb_pe = self.positional_embedding(x_emb)

#         x_encoded_class = self.Tclass(x_emb_pe[:,-2:])  # Exclure les tokens de classe
#         x_encoded_continuous = self.Tcontinuous(x_emb_pe[:,:-2])  # Exclure les tokens de classe

#         x_encoded = torch.cat([x_encoded_continuous, x_encoded_class], dim=-1)  # Concaténer les sorties encodées
#         x_encoded = self.encoder(x_emb_pe)

#         # x_emb = self.patch_embedding(x)
#         # x_emb_pe = self.positional_embedding(x_emb)
        
#         # x_encoded = self.encoder(x_emb_pe)
#         output = self.decoder(x_encoded) #self.pseudo_skip_connection(x) +
#         return output 

class ViTWithDecoder(nn.Module):
    def __init__(self, 
                 input_size,
                 num_patches, 
                 embed_dim, 
                 embedding_type='learnable',
                 dropout=0.1,
                 in_channels=2, 
                 patch_size:int=16,                 
                 num_layers_transformers=4,
                 num_heads=8,
                 mlp_ratio=4.0,
                 attention_dropout=0.1,
                 drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm,
                 activation='gelu',
                 return_attention=False):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size=input_size,
                                              in_channels=in_channels,
                                              embed_dim=embed_dim,
                                              patch_size=patch_size)
        
        self.positional_embedding = PositionalEmbedding(num_patches=num_patches,
                                                        embed_dim=embed_dim,
                                                        embedding_type=embedding_type,
                                                        dropout=dropout)
        
        self.encoder = TransformerEncoder(embed_dim=embed_dim,
                                          num_layers=num_layers_transformers,
                                          num_heads=num_heads,
                                          mlp_ratio=mlp_ratio,
                                          dropout=attention_dropout,
                                          drop_path_rate=drop_path_rate,
                                          norm_layer=norm_layer,
                                          activation=activation,
                                          return_attention=return_attention)
                
        self.decoder = CNNDecoder(patch_size=patch_size,output_size=input_size,embed_dim=embed_dim, num_patches=num_patches)
        
        self.num_skip_connection = min(self.decoder.num_layers, num_layers_transformers)  # Limiter le nombre de connexions résiduelles
        self.pseudo_skip_connection = Skip_Connection(num_skip_connection=self.num_skip_connection,
                                                      embed_dim=embed_dim,
                                                      patch_size=patch_size,
                                                      out_img_size=input_size/2,
                                                      list_in_channels=[self.decoder.upsampling_blocks[-k-1].in_channels for k in range(self.num_skip_connection)]
                                                      )
        # mx num of skip connection is the minimum of both num of layers.
        # pseudo skip connectin

    def forward(self, x):
        x_emb = self.patch_embedding(x)
        x_emb_pe = self.positional_embedding(x_emb)
        x_encoded,list_residuals = self.encoder(x_emb_pe)
        list_residuals = list_residuals[:self.num_skip_connection]  # Prendre les premiers résidus pour les connexions résiduelles
        skip_residuals = self.pseudo_skip_connection(list_residuals)  # Appliquer les connexions résiduelles
        output = self.decoder(x_encoded,skip_residuals) 
        return output




# class TransformerEncoder(nn.Module):
#     def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
#         super().__init__()
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim, nhead=num_heads, 
#             dim_feedforward=embed_dim * 4, 
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#     def forward(self, x):
#         x = self.transformer_encoder(x)
#         return x
    


class VisionTransformer(nn.Module):
    def __init__(self, img_size:int, in_channels:int ,out_channels:int, embed_dim:int, num_heads:int=4, num_layers:int=4, patch_size:int=16, dropout:int=0.1):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embedding = PatchEmbedding(img_size, in_channels, embed_dim, patch_size)
        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, num_layers, dropout)
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


# img_size=256
# in_channels=4
# out_channels=1
# embed_dim=1024
# num_heads=8
# num_layers=4
# patch_size=16
# dropout=0.1


# model = VisionTransformer(
#     img_size=img_size,
#     out_channels=out_channels,
#     in_channels=in_channels,
#     embed_dim=embed_dim,
#     num_heads=num_heads,
#     num_layers=num_layers,
#     patch_size=patch_size,
#     dropout=dropout
# )
# print(model)

# input_tensor = torch.randn(1, in_channels, img_size, img_size)  # Example input
# deb=time.time()
# output = model(input_tensor)
# duration=time.time()-deb
# print("Time taken for forward pass:", duration)
# print("Output shape:", output.shape)  # Should be (1, 1, patch_size, patch_size)


# image_size=256
# patch_size=16
# num_channels=4
# hidden_size=1024
# intermediate_size=4096
# num_attention_heads=8
# num_hidden_layers=8
# hidden_act="gelu"
# layer_norm_eps=1e-12
# dropout_rate=0.1


# # class visionTransformer
# configuration = ViTConfig(
#     image_size=image_size,
#     patch_size=patch_size,
#     num_channels=num_channels,
#     hidden_size=hidden_size,
#     intermediate_size=intermediate_size,
#     num_attention_heads=num_attention_heads,
#     num_hidden_layers=num_hidden_layers,
#     hidden_act=hidden_act,
#     layer_norm_eps=layer_norm_eps,
#     dropout_rate=dropout_rate,
# )

# # Randomly initializing a model (with random weights) from the configuration

# model = ViTModel(configuration)

# # Accessing the model configuration

# configuration = model.config

# # class TimeseriesTransformer
# print(model)
# input_tensor = torch.randn(1, num_channels, img_size, img_size)  # Example input
# deb=time.time()
# output = model(input_tensor)
# n_patches=(image_size//patch_size)**2
# patch_representations=output.last_hidden_state[:,1:,:]
# patch_representations=patch_representations.view(1, n_patches, configuration.hidden_size)
# lineaire=nn.Linear(1024,256)
# output_image=lineaire(patch_representations)
# duration=time.time()-deb
# print("Time taken for forward pass:", duration)
# print("Output shape:", output_image.size())  # Should be (1, 1, patch_size, patch_size)





# Exemple d'utilisation
if __name__ == "__main__":
    if False: #test patch embedding
        # Paramètres d'exemple
        img_size = 256
        in_channels = 3
        embed_dim = 768
        patch_size = 16
        batch_size = 8
        # Création du module de patch embedding
        patch_embedding = PatchEmbedding(img_size, in_channels, embed_dim, patch_size)
        # Simulation d'un tensor d'entrée (image)
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        # Application du patch embedding    
        x_patches = patch_embedding(x)
        print(f"Shape d'entrée: {x.shape}") 
        print(f"Shape après patch embedding: {x_patches.shape}")
    if False: #test positionnal embedding
        # Paramètres d'exemple pour ViT-Base
        num_patches = 256 
        
        
        # Création des embeddings positionnels apprenables
        pos_embed_learnable = PositionalEmbedding(
            num_patches=num_patches,
            embed_dim=embed_dim,
            embedding_type='learnable',
        )
        
        # Création des embeddings positionnels sinusoïdaux
        pos_embed_sinusoidal = PositionalEmbedding(
            num_patches=num_patches,
            embed_dim=embed_dim,
            embedding_type='sinusoidal',
        )
        
        # Simulation d'un tensor d'entrée (patches)
        x = torch.randn(batch_size, num_patches , embed_dim)
        
        # Application des embeddings
        x_with_pos_learnable = pos_embed_learnable(x)
        x_with_pos_sinusoidal = pos_embed_sinusoidal(x)
        
        print(f"Shape d'entrée: {x.shape}")
        print(f"Shape avec pos embedding: {x_with_pos_learnable.shape}")
        print(f"Paramètres apprenables: {sum(p.numel() for p in pos_embed_learnable.parameters())}")
        print(f"Paramètres sinusoïdaux: {sum(p.numel() for p in pos_embed_sinusoidal.parameters())}")
    if False: #test multi-head self attention
        # Paramètres d'exemple
        
        num_heads = 8
        batch_size = 8
        seq_len = 256
        
        # Création du module MultiHeadSelfAttention
        multihead_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        
        # Simulation d'un tensor d'entrée (séquence)
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        # Application de l'auto-attention
        output, attn_weights = multihead_attn(x)
        
        print(f"Shape d'entrée: {x.shape}")
        print(f"Shape après auto-attention: {output.shape}")
        print(f"Shape des poids d'attention: {attn_weights.shape}")
    if False: #test transformer encoder

        # Paramètres d'exemple
        embed_dim = 768
        num_heads = 12
        num_layers = 4
        batch_size = 8
        seq_len = 256
        dropout=0.1
        mlp_ratio = 4.
        drop_path_rate=0.0
        norm_layer=nn.LayerNorm
        activation='gelu'
        return_attention=False
        # Création du module Transformer Encoder
        transformer_encoder = TransformerEncoder(
                                embed_dim=embed_dim,
                                num_layers=num_layers,
                                num_heads=num_heads,
                                dropout=dropout,
                                mlp_ratio = mlp_ratio,
                                drop_path_rate=drop_path_rate,
                                norm_layer=norm_layer,
                                activation=activation,
                                return_attention=return_attention)
        
        # Simulation d'un tensor d'entrée (séquence)
        x = torch.randn(batch_size, seq_len, embed_dim)
        # Application de l'encodeur Transformer
        output = transformer_encoder(x)
        print(f"Shape d'entrée: {x.shape}")
        print(f"Shape après Transformer Encoder: {output.shape}")
    if False: #test decoder block

        patch_size=16
        output_size=256

        embded_dim=1024
        num_patches=256
        
        model=CNNDecoder(patch_size,output_size,embded_dim, num_patches)
        x= torch.randn(1, num_patches, embded_dim)
        output=model(x)
        print(f"Shape d'entrée: {x.shape}")
        print(f"Shape après CNN Decoder: {output.shape}")
    if True: #test ViT with decoder
        img_size = 256
        in_channels = 10
        out_channels = 1
        embed_dim = 768
        num_heads = 12
        num_layers = 3
        patch_size = 16
        dropout = 0.1
        num_patches = (img_size // patch_size) ** 2
        
        model = ViTWithDecoder(
            input_size=img_size,
            num_patches=num_patches,
            embed_dim=embed_dim,
            embedding_type='sinusoidal',
            dropout=dropout,
            in_channels=in_channels,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=4.0,
            attention_dropout=0.1,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            activation='gelu',
            return_attention=False
        )
        
        # Simulation d'un tensor d'entrée (image)
        x = torch.randn(1, in_channels, img_size, img_size)
        
        # Application du modèle ViT avec décodeur
        t=time.time()
        output = model(x)
        print(f"Temps d'exécution: {time.time()-t:.2f} secondes")
        print(f"Paramètres : {sum(p.numel() for p in model.parameters())}")
        print(f"Shape d'entrée: {x.shape}")
        print(f"Shape après ViT avec décodeur: {output.shape}")












