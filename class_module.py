import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



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
        def __init__(self, in_channels, out_channels, skip_channels):
            super(UpsamplingBlock, self).__init__()
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_block1 = ConvBlock(in_channels + skip_channels, in_channels)
            self.conv_block2 = ConvBlock(in_channels, in_channels)
            self.conv_block3 = ConvBlock(in_channels, out_channels)

        def forward(self, x, skip_connection):
            x = torch.cat((x, skip_connection), dim=1)
            x = self.conv_block1(x)
            x = self.upsample(x)
            
            
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            return x

