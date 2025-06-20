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

from NewBase import Base
from SwinTransformers import SwinTranformers,Swin_UperNet
from middle_fusion import Middle_fusion_en as mf_
import pytorch_lightning as pl

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar
from class_module import *

class TransformerArchitectures(Base):
    def __init__(self, params):
        # init base
        super(TransformerArchitectures, self).__init__(params)
        
        # reorganize sources values
        source_order = ['rgb', 'hs', 'dtm', 'sar','lc','sau','esa']
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
                if source == 'esa':
                    input_channels = input_channels +self.conf['num_class_esa']  # should be 10
        
        
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


        elif self.conf['method'] == 'swin_transformers':
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
                if source == 'esa':
                    input_channels = input_channels +self.conf['num_class_esa']
            
            self.net=Swin_UperNet(num_channels=input_channels)

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

        elif self.conf['method'] == 'swin_transformers':
            return self.net(batch)

    def create_transform_function(self, transform_list):
        # create transformation function
        def transform_inputs(inps):
            # create transformation
            sources_possibles = ['rgb', 'hs', 'dtm', 'sar','lc','sau','esa', 'ndvi']
            inps_dict = {source: inps[i] for i, source in enumerate(self.conf['sources']+['ndvi'])}  # add ndvi to the inputs dict

            # Checking if all keys have a designated value else 0 TODO can maybe be improve to reduce storage and calculations
            # Ensure all sources have a value, defaulting to a zero tensor
            inps_dict = {source: inps_dict.get(source, torch.zeros_like(next(iter(inps_dict.values())))) for source in sources_possibles}
            
            # Extract values for required sources
            rgb, hs, dtm, sar, lc, sau, esa, ndvi = (inps_dict.get(key) for key in sources_possibles)


            normalize_rgb, normalize_hs, normalize_dtm, normalize_sar, transforms_augmentation = transform_list
            #no normalization for ndvi because it is already between -1 and 1
            ndvi=ndvi.unsqueeze(2) # so ndvi has the same shape as the others
            # no normalization for lc, sau,esa because it is onehot encoded


            rgb = (rgb.numpy() - self.loaded_min_dict_before_normalization['rgb']) / (self.loaded_max_dict_before_normalization['rgb'] - self.loaded_min_dict_before_normalization['rgb'])
            hs = (hs.numpy() - self.loaded_min_dict_before_normalization['hs']) / (self.loaded_max_dict_before_normalization['hs'] - self.loaded_min_dict_before_normalization['hs'])
            dtm = (dtm.numpy() - self.loaded_min_dict_before_normalization['dtm']) / (self.loaded_max_dict_before_normalization['dtm'] - self.loaded_min_dict_before_normalization['dtm'])
            sar = (sar.numpy() - self.loaded_min_dict_before_normalization['sar']) / (self.loaded_max_dict_before_normalization['sar'] - self.loaded_min_dict_before_normalization['sar'])
            #no need to normalize the ndvi because it is already between -1 and 1 and lc,sau are onehot encoded
            ndvi = ndvi.numpy()
            lc=lc.numpy()
            sau=sau.numpy()
            esa=esa.numpy()

            rgb = rgb.astype(np.float32)
            hs = hs.astype(np.float32)
            dtm = dtm.astype(np.float32)
            sar = sar.astype(np.float32)
            ndvi = ndvi.astype(np.float32)
            lc = lc.astype(np.float32)
            sau = sau.astype(np.float32)
            esa = esa.astype(np.float32)
            
            rgb = normalize_rgb(image=rgb)['image']
            hs = normalize_hs(image=hs)['image']
            dtm = normalize_dtm(image=dtm)['image']
            sar = normalize_sar(image=sar)['image']


            # initialize the transforms
            transforms = A.Compose([transforms_augmentation], is_check_shapes=False, 
                                    additional_targets={key: 'image' for key in sources_possibles[1:]}) # rgb is not part of the additinnal target because it is the reference for size
            # apply the transforms
            sample = transforms(image=rgb,
                                hs=hs,
                                dtm=dtm,
                                sar=sar,
                                lc=lc,
                                sau=sau,
                                esa=esa,
                                ndvi=ndvi
                                
                                )
            # get images
            rgb, hs, dtm, sar, lc, sau, esa, ndvi = sample['image'],*(sample[key] for key in sources_possibles[1:])

            outputs_dict = {key: locals()[key] for key in sources_possibles}
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
        return self.val_transforms()



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



class Transformerswithdetails(nn.Module):
    def __init__(self, 
                 input_size,
                 num_patches, 
                 embed_dim, 
                 embedding_type='sinusoidal',
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
        # Save input parameters
        self.input_size = input_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.embedding_type = embedding_type
        self.dropout = dropout
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_layers_transformers = num_layers_transformers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attention_dropout = attention_dropout
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.activation = activation
        self.return_attention = return_attention

        
        self.patch_embedding = PatchEmbedding(img_size=input_size,
                                              in_channels=in_channels,
                                              embed_dim=embed_dim,
                                              patch_size=patch_size)
        self.details_embedding=PatchEmbedding(img_size=input_size,
                                              in_channels=in_channels,
                                              embed_dim=embed_dim,
                                              patch_size=patch_size)
        
        self.positional_embedding = PositionalEmbedding(num_patches=num_patches,
                                                        embed_dim=embed_dim,
                                                        embedding_type=embedding_type,
                                                        dropout=dropout)

        self.details_encoder=TransformerEncoder(embed_dim=embed_dim,
                                                num_layers=4,
                                                num_heads=4,
                                                mlp_ratio=4,
                                                return_attention=False
                                                )
        
        self.encoder=TransformerEncoder(embed_dim=embed_dim,
                                                num_layers=num_layers_transformers,
                                                num_heads=num_heads,
                                                mlp_ratio=mlp_ratio,
                                                return_attention=False
                                                )

        self.decoder= nn.Sequential(UpsamplingBlock(2*embed_dim,embed_dim//2),
                                    UpsamplingBlock(embed_dim//2,embed_dim//8),
                                    UpsamplingBlock(embed_dim//8,embed_dim//16),
                                    UpsamplingBlock(embed_dim//16,1))
                                 



    def forward(self,x):
        
        details=calculate_tensor_gradient(x)

        x_emb = self.patch_embedding(x)
        details_emb=self.details_embedding(details)
        
        x_emb_pe = self.positional_embedding(x_emb)
        details_emb_pe = self.positional_embedding(details_emb)

        details_encoded,_=self.details_encoder(details_emb_pe)
        x_encoded,_=self.encoder(x_emb_pe)

        x=torch.cat((x_encoded,details_encoded),dim=2) #fusion of values features and geographic details
        x=x.permute(0,2,1)
        x=x.view(x.size(0),x.size(1),self.patch_size,self.patch_size)

        x=self.decoder(x)

        return x


def calculate_tensor_gradient(image_tensor):
    """
    Version avec normalisation globale sur tout le batch
    """
    batch_size = image_tensor.size(0)
    num_channels = image_tensor.size(1)
    device = image_tensor.device
    dtype = image_tensor.dtype
    
    # Définir les noyaux de Sobel
    sobel_x = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).expand(num_channels, 1, 3, 3)
    
    sobel_y = torch.tensor([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).expand(num_channels, 1, 3, 3)

    
    # Calculer les gradients
    grad_x = F.conv2d(image_tensor, sobel_x, padding=1, groups=num_channels)
    grad_y = F.conv2d(image_tensor, sobel_y, padding=1, groups=num_channels)
    
    # Calculer la magnitude du gradient
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    
    # Normaliser la magnitude du gradient (globalemnt sur tout le tenseur)
    grad_min = grad_magnitude.min()
    grad_max = grad_magnitude.max()
    
    if (grad_max - grad_min) != 0:
        grad_magnitude = (grad_magnitude - grad_min) / (grad_max - grad_min)
    
    return grad_magnitude

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

'''
class Transformer_latefusion(nn.Module):
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

        self.patch_embedding=PatchEmbedding(img_size=input_size,in_channels=in_channels,embed_dim=embed_dim,patch_size=patch_size)

        self.positionnal_embedding=PositionalEmbedding(num_patches=num_patches,embed_dim=embed_dim,embedding_type=embedding_type)

        self.encoder=nn.ModuleDict()
        for 

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
'''  


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
    if False: #test ViT with decoder
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
    if False: # test de ReversePatchEmbedding
        out_img_size = 32
        embed_dim = 1024
        out_channels = 7
        patch_size=16
        num_patches=256
        test=ReversePatchEmbedding(out_img_size=out_img_size,out_channels=out_channels,embed_dim=embed_dim,patch_size=patch_size)
        x = torch.randn(1, embed_dim,num_patches)  # Simuler un tensor d'entrée
        output = test(x)
        print(f"Shape d'entrée: {x.shape}")
        print(f"Shape après ReversePatchEmbedding: {output.shape}")  # Devrait être (1, in_channels, img_size, img_size)

    if False : #test de skip connection

    # Paramètres d'exemple

        embed_dim = 1024
        patch_size=16
        input_size=256
        num_patches=256
        
        encoder = TransformerEncoder(embed_dim=embed_dim,
                                          num_layers=4,
                                          num_heads=2,
                                          mlp_ratio=2,
                                          dropout=0,
                                          drop_path_rate=0,
                                          activation='gelu',
                                          return_attention=False)
               
        decoder = CNNDecoder(patch_size=patch_size,output_size=input_size,embed_dim=embed_dim, num_patches=num_patches)
        
        num_skip_connection=min(decoder.num_layers, encoder.num_layers)
        x= [torch.randn(1, embed_dim,num_patches) for _ in range(num_skip_connection) ]  # Simuler un tensor d'entrée
        pseudo_skip_connection = Skip_Connection(num_skip_connection=num_skip_connection,
                                                      embed_dim=embed_dim,
                                                      list_in_channels=[decoder.upsampling_blocks[-k-1].in_channels for k in range(num_skip_connection)]
                                                      )

        res= pseudo_skip_connection(x)
        print(f"Shape après Skip Connection: {[ri.shape for ri in res]}")  # Liste de résidus transformés
        
        decoder(torch.randn(1, embed_dim,num_patches ), res)  
    if False:
        
        input_size=256
        num_patches=256 
        embed_dim=768 
        embedding_type='sinusoidal'
        dropout=0.1
        in_channels=10
        patch_size=16               
        num_layers=4
        num_heads=8
        mlp_ratio=4.0
        attention_dropout=0.1
        drop_path_rate=0.0
        norm_layer=nn.LayerNorm
        activation='gelu'
        return_attention=False

        patch_embedding = PatchEmbedding(img_size=input_size,
                                              in_channels=in_channels,
                                              embed_dim=embed_dim,
                                              patch_size=patch_size)
        
        positional_embedding = PositionalEmbedding(num_patches=num_patches,
                                                        embed_dim=embed_dim,
                                                        embedding_type=embedding_type,
                                                        dropout=dropout)
        
        encoder = TransformerEncoder(embed_dim=embed_dim,
                                          num_layers=num_layers,
                                          num_heads=num_heads,
                                          mlp_ratio=mlp_ratio,
                                          dropout=attention_dropout,
                                          drop_path_rate=drop_path_rate,
                                          norm_layer=norm_layer,
                                          activation=activation,
                                          return_attention=return_attention)
                
        decoder = CNNDecoder(patch_size=patch_size,output_size=input_size,embed_dim=embed_dim, num_patches=num_patches)
        
        num_skip_connection = min(decoder.num_layers, num_layers)  # Limiter le nombre de connexions résiduelles
        pseudo_skip_connection = Skip_Connection(num_skip_connection=num_skip_connection,
                                                      embed_dim=embed_dim,
                                                      list_in_channels=[decoder.upsampling_blocks[-k-1].in_channels for k in range(num_skip_connection)]
                                                      )
        input= torch.randn(1, in_channels, input_size, input_size)  # Simuler un tensor d'entrée
        x_emb = patch_embedding(input)
        x_pos= positional_embedding(x_emb)
        x,list_residuals = encoder(x_pos)
        skip_residuals = pseudo_skip_connection(list_residuals) 
        output = decoder(x, skip_residuals)
        print(f"Shape d'entrée: {input.shape}")
        print(f"Shape après le modèle: {output.shape}")  # Devrait être (1, 1, input_size, input_size)

    if True: # test de TransformerArchitecure
        import json
        with open('Tparams.json', 'r') as f:
            conf = json.load(f)

        model= TransformerArchitectures(conf)
        callbacks = [
            RichProgressBar(),
            ModelCheckpoint(
                monitor='val_mae',
                mode='min',
                save_top_k=1,
                save_last=True,
                filename='l1_loss-{epoch:02d}-{val_loss:.3f}'
            ),
            LearningRateMonitor(logging_interval='epoch'),
            EarlyStopping(
                monitor='val_loss',
                min_delta=0.001,
                patience=20,
                verbose=True,
                mode='min',
                strict=True,
            )
        ]

        # Spécifiez le chemin vers le fichier de checkpoint

        # Créez le Trainer
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=1,
            num_sanity_val_steps=2,
            logger=TensorBoardLogger('checkpoints', name='test'+conf['experiment_name'] + "_" + "_".join(conf['sources'] + [conf['method']])),
            callbacks=callbacks
        )

        print("Training...")
        print(conf['sources'])
        print(conf['method'])
        trainer.fit(model)

        # Testez le modèle
        trainer.test(model)











