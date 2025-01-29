import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder: First block of the U-Net model
        self.enc1 = self._block(in_channels, 64)
        self.pool = nn.MaxPool2d(2)  # Max pooling layer to downsample
        
        # Additional layers can be added to the encoder...

    def _block(self, in_ch, out_ch):
        """
        A helper function to create a block consisting of:
        - Two convolutional layers with ReLU activations
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # Convolution layer
            nn.ReLU(),  # ReLU activation
            nn.Conv2d(out_ch, out_ch, 3, padding=1),  # Second convolution layer
            nn.ReLU()  # ReLU activation
        )
    
    def forward(self, x):
        """
        The forward pass function which processes the input image `x`.
        In this simplified version, it just returns the input `x`.
        """
        x = self.enc1(x)  # Pass input through the encoder
        x = self.pool(x)  # Downsample using max pooling
        return x  # Return the processed mask/output (in practice, more steps like decoding should follow)
