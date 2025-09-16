from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

UPSAMPLING_MODES = ['nearest', 'bilinear', 'pixel_shuffle', 'conv_transpose']

class SrCnn(nn.Module):
    def __init__(self, scale: int,  in_channels: int, channels: int, kernel_size: int = 3, stride: int = 1,
                 activation: Optional[nn.Module] = nn.ReLU, mode='bilinear'):
        """ Super Resolution CNN:  -> conv -> activation

        Args:
            scale: (int) upscale factor.
            in_channels: (int) input channels.
            channels: (int) number of output channels for the convolution op.
            kernel_size: (int) kernel size for the convolution op.
            stride: (int) stride for the convolution op.
            activation: (int) activation after the convolution op (optional).
        """
        super().__init__()

        self.conv = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride, padding='same', padding_mode='replicate')
        self.upsample = nn.Upsample(scale_factor=scale, mode=mode)
        self.activation = activation() if activation is not None else nn.Identity()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.activation(x)

        return x
