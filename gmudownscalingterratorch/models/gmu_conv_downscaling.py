from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

UPSAMPLING_MODES = ['nearest', 'bilinear', 'pixel_shuffle', 'conv_transpose']

class PixelShuffleBlock(nn.Module):
    def __init__(self, scale: int, in_channels: int, channels: int, kernel_size: int = 3, stride: int = 1,
                 activation: Optional[nn.Module] = nn.PReLU):
        """ Upsampling with PixelShuffle: conv -> shuffle -> activation

        Args:
            scale: (int) upscale factor.
            in_channels: (int) input channels.
            channels: (int) number of output channels.
            kernel_size: (int) kernel size for the convolution op.
            stride: (int) stride for the convolution op.
            activation: (int) activation after the convolution op (optional).
        """
        super().__init__()

        self.conv = nn.Conv2d(in_channels, channels * scale ** 2,
                              kernel_size=kernel_size, stride=stride, padding='same', padding_mode='replicate')
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.activation = activation() if activation is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.activation(x)

        return x


class InterpBlock(nn.Module):
    def __init__(self, scale: int,  in_channels: int, channels: int, kernel_size: int = 3, stride: int = 1,
                 activation: Optional[nn.Module] = nn.ReLU, mode='bilinear'):
        """ Upsampling with interpolation: interpolation -> conv -> activation

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


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, kernel_size: int = 2, stride: int = 2,
                 activation: Optional[nn.Module] = nn.ReLU):
        """ Upsampling with ConvTranspose2d: conv -> activation
            (stride must be > 1 to increase spatial size)

        Args:
            in_channels: (int) input channels.
            channels: (int) number of output channels for the convolution op.
            kernel_size: (int) kernel size for the convolution op.
            stride: (int) stride for the convolution op.
            activation: (int) activation after the convolution op (optional).
        """
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_channels, channels, kernel_size=kernel_size, stride=stride)
        self.activation = activation() if activation is not None else nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.activation(x)

        return x


class ConvEncoderDecoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            channels: int,
            out_channels: int,
            scale: List[int],
            upsampling_mode: str,
            kernel_size: List[int],
            stride: int = 1,
            activation: Optional[nn.Module] = nn.PReLU,
        ):
        """ Simple convolutional decoder for downscaling.

        Args:
            scale: list with upscale factors for each stage.
            in_channels: (int) input channels.
            channels: (int) number of channels for the convolution ops.
            upsampling_mode: (str) upsampling to use ('nearest', 'bilinear', 'pixel_shuffle', 'conv_transpose').
            kernel_size: (int) kernel size for the convolution ops.
            stride: (int) stride for the convolution ops.
            activation: (int) activation after each convolution ops (optional).
            out_channels: (int) number of output channels (target vars). Default = 1.
        """

        super().__init__()

        self.scale = scale
        self.kernel_size = kernel_size 
        if upsampling_mode not in UPSAMPLING_MODES:
            raise ValueError(f"Incorrect upsampling mode {upsampling_mode}.")
        self.upsampling_mode = upsampling_mode
        
        # build layers
        current_ch = in_channels
        self.blocks = nn.ModuleList()
        for k_i, s_i in zip(self.kernel_size, self.scale):
            if upsampling_mode == 'pixel_shuffle':
                layer = PixelShuffleBlock(scale=s_i, in_channels=current_ch, channels=channels,
                                          kernel_size=k_i, stride=stride, activation=activation)
            elif upsampling_mode == 'conv_transpose':
                layer = ConvTransposeBlock(in_channels=current_ch, channels=channels,
                                           kernel_size=k_i, stride=s_i, activation=activation)
            else:
                layer = InterpBlock(scale=s_i, in_channels=current_ch, channels=channels, kernel_size=k_i,
                                    stride=stride, activation=activation, mode=upsampling_mode)
            current_ch = channels

            self.blocks.append(layer)

        # Assuming downscaling of 1 variable
        self.out_conv = nn.Conv2d(in_channels=current_ch, out_channels=out_channels,
                                  kernel_size=3, stride=1, padding='same', padding_mode='replicate')

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: Input of shape (batch, in_channels, lat, lon)
        Returns:
            Tensor of shape (batch, out_channels, scale*lat, scale*lon)
        '''

        for block in self.blocks:
            x = block(x)
        x = self.out_conv(x)

        return x
