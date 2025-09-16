import os
from typing import Any

import kornia.augmentation as K
import torch
import logging
from torchgeo.trainers import BaseTask

logger = logging.getLogger(__name__)

class GmuWxcDownscaleMerra2PrismTask(BaseTask):
    def __init__(
            self,
            input_size_time: int = 2,
            in_channels_static: int = 120
            ):
        """
        in_channels: int,
        input_size_time: int,
        in_channels_static: int,
        input_scalers_mu: Tensor,
        input_scalers_sigma: Tensor,
        input_scalers_epsilon: float,
        static_input_scalers_mu: Tensor,
        static_input_scalers_sigma: Tensor,
        static_input_scalers_epsilon: float,
        output_scalers: Tensor,
        n_lats_px: int,
        n_lons_px: int,
        patch_size_px: tuple[int],
        mask_unit_size_px: tuple[int],
        mask_ratio_inputs: float,
        mask_ratio_targets: float,
        embed_dim: int,
        n_blocks_encoder: int,
        n_blocks_decoder: int,
        mlp_multiplier: float,
        n_heads: int,
        dropout: float,
        drop_path: float,
        parameter_dropout: float,
        residual: str,
        masking_mode: str,
        positional_encoding: str,
        encoder_shifting: bool = False,
        decoder_shifting: bool = False,
        checkpoint_encoder: list[int]=[],
        checkpoint_decoder: list[int]=[],
                    
        Args:
            in_channels: Number of input channels.
            input_size_time: Number of timestamps in input.
            in_channels_static: Number of input channels for static data.
            input_scalers_mu: Tensor of size (in_channels,). Used to rescale
                input.
            input_scalers_sigma: Tensor of size (in_channels,). Used to rescale
                input.
            input_scalers_epsilon: Float. Used to rescale input.
            static_input_scalers_mu: Tensor of size (in_channels_static). Used
                to rescale static inputs.
            static_input_scalers_sigma: Tensor of size (in_channels_static).
                Used to rescale static inputs.
            static_input_scalers_epsilon: Float. Used to rescale static inputs.
            output_scalers: Tensor of shape (in_channels,). Used to rescale
                output.
            n_lats_px: Total latitudes in data. In pixels.
            n_lons_px: Total longitudes in data. In pixels.
            patch_size_px: Patch size for tokenization. In pixels lat/lon.
            mask_unit_size_px: Size of each mask unit. In pixels lat/lon.
            mask_ratio_inputs: Masking ratio for inputs. 0 to 1.
            mask_ratio_targets: Masking ratio for targets. 0 to 1.
            embed_dim: Embedding dimension
            n_blocks_encoder: Number of local-global transformer pairs in
                encoder.
            n_blocks_decoder: Number of local-global transformer pairs in
                decoder.
            mlp_multiplier: MLP multiplier for hidden features in feed forward
                networks.
            n_heads: Number of attention heads.
            dropout: Dropout.
            drop_path: DropPath.
            parameter_dropout: Dropout applied to parameters.
            residual: Indicates whether and how model should work as residual
                model. Accepted values are 'climate', 'temporal' and 'none'
            positional_encoding: possible values are ['absolute' (default), 'fourier'].
                'absolute'  lat lon encoded in 3 dimensions using sine and cosine
                'fourier' lat/lon to be encoded using various frequencies
            masking_mode: String ['local', 'global', 'both'] that controls the
                type of masking used.
            checkpoint_encoder: List of integers controlling if gradient checkpointing is used on encoder.
                Format: [] for no gradient checkpointing. [3, 7] for checkpointing after 4th and 8th layer etc.
            checkpoint_decoder: List of integers controlling if gradient checkpointing is used on decoder.
                Format: See `checkpoint_encoder`.
            masking_mode: The type of masking to use {'global', 'local', 'both'}
            encoder_shifting: Whether to use swin shifting in the encoder.
            decoder_shifting: Whether to use swin shifting in the decoder.

        """
        super().__init__()

    def configure_models(self):
        input_size_time: int = self.hparams['input_size_time']
        in_channels_static: int = self.hparams['in_channels_static']
        print("input_size_time=", input_size_time)
        return super().configure_models()