import torch.nn as nn

class ConvStem(nn.Module):
    """
    This convolution "stem" is essentially a small stack of convolutional layers that you can run on the 
    raw image *before* breaking it up into patches. 
    
    Provides: 
        - early feature extraction 
        - overlapping receptive fields
        - downsampling control
        - stronger inductive bias.
    """
    def __init__(self, in_channels: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim // 2, kernel_size = 2, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(hidden_dim // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # spatial size: H, W -> H/2, W/2
        self.conv2 = nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size = 2, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2) # spatial size: H/2, W/2 -> H/4, W/4
        # if skipping from after pool1 to after conv2-block, adjust dims
        self.skip = nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=2, stride=2, bias=False) # 1x1 conv to match channels and downsample
                                  
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, H, W)

        Returns:
            Tensor of shape (batch_size, hidden_dim, H/4, W/4)
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        skip = self.skip(out)  # Save skip after first pooling

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = out + skip  # Add the skip connection
        return out