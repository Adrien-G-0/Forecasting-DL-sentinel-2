import torch
import torch.nn as nn

class ConvBlock_mf(nn.Module):
    """
    Convolutional block with multiple layers for multi-modal fusion.

    This block creates a sequence of convolutional layers followed by ReLU activations.
    The architecture is defined by the 'channels' and 'kernels' parameters in the configuration.

    Args:
        conf (dict): Configuration dictionary containing:
            - 'channels': List of channel dimensions for each layer (input and output)
            - 'kernels': List of kernel sizes for each convolutional layer
    """
    def __init__(self, conf):
        """
        Initialize the convolutional block based on the configuration.

        Args:
            conf (dict): Configuration dictionary specifying the architecture.
        """
        super(ConvBlock_mf, self).__init__()
        channels = conf['channels']
        kernels = conf['kernels']

        # Initialize an empty list to store the layers
        layers = []
        # Loop through each kernel size and create corresponding convolutional layer
        for i in range(len(kernels)):
            # Add a convolutional layer with specified kernel size and padding
            layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=kernels[i], padding=kernels[i]//2, stride=1))
            # Add ReLU activation after each convolutional layer
            layers.append(nn.ReLU(inplace=True))

        # Combine all layers into a sequential module
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the convolutional block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Output tensor after passing through all convolutional layers
        """
        # Apply the sequential convolutional layers to the input
        return self.conv(x)

class Middle_fusion_en(nn.Module):
    """
    Middle Fusion module that combines features from multiple sources.

    This module applies separate convolutional processing to each input source
    and then concatenates the resulting features along the channel dimension.

    Args:
        conf (dict): Configuration dictionary containing:
            - 'sources': List of data source types to use (must be from supported sources)
            - Configuration for each potential source (conf_rgb, conf_dtm, etc.)
    """
    def __init__(self, conf):
        """
        Initialize the middle fusion module.

        Args:
            conf (dict): Configuration dictionary specifying the architecture and sources.
        """
        super(Middle_fusion_en, self).__init__()
        self.sources = conf["sources"]
        self.conv = nn.ModuleDict()

        # Store configurations for each potential source
        # These configurations specify the architecture of the convolutional blocks
        # for each data source type
        self.conf_rgb = conf["conf_rgb"]
        self.conf_dtm = conf["conf_dtm"]
        self.conf_sar = conf["conf_sar"]
        self.conf_hs = conf["conf_hs"]
        self.conf_lc = conf["conf_lc"]
        self.conf_sau = conf["conf_sau"]
        self.conf_esa = conf["conf_esa"]

        # Validate that all sources are from the supported list
        if not all(source in ['rgb', 'hs', 'dtm', 'sar', 'lc', 'sau', 'esa'] for source in self.sources):
            raise Exception("Invalid sources, must be in ['rgb', 'hs', 'dtm', 'sar','lc','sau','esa' ]")

        # Validate configuration dimensions for each potential source
        # The channels list should have length = kernels length + 1
        # because each convolutional layer consumes one input channel and produces one output channel,
        # and the length difference accounts for the input dimension

        # Check RGB configuration
        if len(self.conf_rgb['channels']) != len(self.conf_rgb['kernels']) + 1:
            raise Exception("RGB configurations are wrong, channels length must be equal to kernels length + 1")

        # Check HS configuration
        if len(self.conf_hs['channels']) != len(self.conf_hs['kernels']) + 1:
            raise Exception("HS configurations are wrong, channels length must be equal to kernels length + 1")

        # Check SAR configuration
        if len(self.conf_sar['channels']) != len(self.conf_sar['kernels']) + 1:
            raise Exception("SAR configurations are wrong, channels length must be equal to kernels length + 1")

        # Check DTM configuration
        if len(self.conf_dtm['channels']) != len(self.conf_dtm['kernels']) + 1:
            raise Exception("DTM configurations are wrong, channels length must be equal to kernels length + 1")

        # Check LC configuration
        if len(self.conf_lc['channels']) != len(self.conf_lc['kernels']) + 1:
            raise Exception("LC configurations are wrong, channels length must be equal to kernels length + 1")

        # Check SAU configuration
        if len(self.conf_sau['channels']) != len(self.conf_sau['kernels']) + 1:
            raise Exception("SAU configurations are wrong, channels length must be equal to kernels length + 1")

        # Check ESA configuration
        if len(self.conf_esa['channels']) != len(self.conf_esa['kernels']) + 1:
            raise Exception("ESA configurations are wrong, channels length must be equal to kernels length + 1")

        # Create convolutional blocks for each specified source
        # Each source gets its own convolutional block with its specific configuration
        for source in self.sources:
            if source == 'rgb':
                self.conv[source] = ConvBlock_mf(self.conf_rgb)
            elif source == 'hs':
                self.conv[source] = ConvBlock_mf(self.conf_hs)
            elif source == 'sar':
                self.conv[source] = ConvBlock_mf(self.conf_sar)
            elif source == 'dtm':
                self.conv[source] = ConvBlock_mf(self.conf_dtm)
            elif source == 'lc':
                self.conv[source] = ConvBlock_mf(self.conf_lc)
            elif source == 'sau':
                self.conv[source] = ConvBlock_mf(self.conf_sau)
            elif source == 'esa':
                self.conv[source] = ConvBlock_mf(self.conf_esa)

    def forward(self, inputs):
        """
        Forward pass of the middle fusion module.

        Args:
            inputs (list or tuple of torch.Tensor): Input tensors corresponding to each source.
                The order of tensors should match the order of sources specified in the configuration.

        Returns:
            torch.Tensor: Concatenated features from all sources along the channel dimension.
        """
        # Check that the number of inputs matches the number of sources
        if len(inputs) != len(self.sources):
            raise Exception(f"Expected {len(self.sources)} inputs, got {len(inputs)}")

        features = []
        # Process each input with its corresponding convolutional block
        for i, source in enumerate(self.sources):
            feature = self.conv[source](inputs[i])
            features.append(feature)

        # Concatenate all features along the channel dimension
        return torch.cat(features, dim=1)

    



if __name__ == '__main__':
        
        import json
        with open('params.json') as f:
            conf = json.load(f)
        model=Middle_fusion_en(conf)
        print(model)
        inputa_rgb=torch.randn(1,3,256,256)
        inputa_hs=torch.randn(1,182,256,256)
        inputa_dtm=torch.randn(1,1,256,256)
        inputa_sar=torch.randn(1,2,256,256)
        inputa_lc=torch.randn(1,8,256,256)
        inputa_sau=torch.randn(1,10,256,256)
        inputs=[inputa_dtm, inputa_sar,inputa_lc,inputa_sau]
        output=model(inputs)
        print(output.shape)