## Modalities Used

- SAR (Synthetic Aperture Radar)
- Land Cover
- Soil Agriculture Use
- ESA WorldCover

Partially at the beginning
- DTM (Digital Terrain Model)
- RGB (Red, Green, Blue)
- Hyperspectral


# Neural Network Model Architectures

## CNN: Unet

### Early Fusion
- **Encoder**:
  - Convolution: input_channels -> 64
  - Convolution: 64 -> 128
  - Convolution: 128 -> 256
  - Convolution: 256 -> 512
- **Bottleneck**: Two sequential convolutions (512 -> 1024 -> 512)
- **Decoder**:
  - Upsampling: 512 -> 256
  - Upsampling: 256 -> 128
  - Upsampling: 128 -> 64
  - Upsampling: 64 -> 1
- **Activation**: Sigmoïd
- **Skip Connections**: Between encoder and decoder

### Middle Fusion
- **Embedding Dimension**: (64, 32, 16)
- **Initial Convolution**: input_channels -> Embedding Dimension
- **Concatenation**: Fusion of embeddings
- **Encoder**:
  - Convolution: Embedding Dimension * number of inputs -> 64
  - Convolution: 64 -> 128
  - Convolution: 128 -> 256
  - Convolution: 256 -> 512
- **Bottleneck**: Two sequential convolutions (512 -> 1024 -> 512)
- **Decoder**:
  - Upsampling: 512 -> 256
  - Upsampling: 256 -> 128
  - Upsampling: 128 -> 64
  - Upsampling: 64 -> 1
- **Activation**: Sigmoïd
- **Skip Connections**: Between encoder and decoder

### Late Fusion
- **Encoder per Modality**:
  - Convolution: input_channels -> 2 * input_channels
  - Convolution: 2 * input_channels -> 4 * input_channels
  - Convolution: 4 * input_channels -> 8 * input_channels
  - Convolution: 8 * input_channels -> 16 * input_channels
- **Size**: Reduction from 256 to 16
- **Concatenation**: Fusion of outputs and residuals
- **Decoder**:
  - Upsampling: input_channels -> input_channels // 2
  - Upsampling: input_channels // 2 -> input_channels // 4
  - Upsampling: input_channels // 4 -> input_channels // 8
  - Upsampling: input_channels // 8 -> 1
- **Activation**: Sigmoïd

## Transformer

### Encoding Part

- **Basic Transformer**
  - **Architecture**: Encoder-decoder with self-attention mechanisms.
  - **Layers**: Multi-head attention and feed-forward networks.

- **Vision Transformer (ViT)**
  - **Architecture**: Divides images into patches, treats each patch as a token.
  - **Layers**: Self-attention to capture dependencies between patches.
  - **Usage**: Image classification and computer vision tasks.

- **Swin Transformers**
  - **Architecture**: Uses shifted windows to compute local attention.
  - **Layers**: Shifted windows to capture dependencies at different scales.
  - **Usage**: Computer vision tasks requiring hierarchical modeling.

### Decoding Part

- **CNN with Skip Connections**  ***(also tested with contour details in skip connections)***
  - **Architecture**: Convolutional neural network with residual connections.
  - **Layers**: Convolutions and upsampling with skip connections for improved reconstruction.
  - **Usage**: Image segmentation and data reconstruction.

- **Upernet**
  - **Architecture**: Pyramid Scene Parsing network with a Upernet decoder.
  - **Layers**: Feature Pyramid Network (FPN) and Pyramid Pooling Module (PPM).
  - **Usage**: Semantic segmentation to capture multi-scale details.
