import os
import torch
import yaml
from typing import LiteralString
from torch import Tensor
import logging
from torchgeo.trainers import BaseTask
from PrithviWxC.model import PrithviWxC
from huggingface_hub import hf_hub_download
from PrithviWxC.dataloaders.merra2 import (
    input_scalers,
    output_scalers,
    static_input_scalers
)

from gmudownscalingterratorch.models import(
    
)

logger = logging.getLogger(__name__)


class GmuPrithviWxcDownscalingTask(BaseTask):
    valid_surface_vars = [
        "EFLUX",
        "GWETROOT",
        "HFLUX",
        "LAI",
        "LWGAB",
        "LWGEM",
        "LWTUP",
        "PS",
        "QV2M",
        "SLP",
        "SWGNT",
        "SWTNT",
        "T2M",
        "TQI",
        "TQL",
        "TQV",
        "TS",
        "U10M",
        "V10M",
        "Z0M",
    ]
    valid_static_surface_vars = ["FRACI", "FRLAND", "FROCEAN", "PHIS"]
    valid_vertical_vars = ["CLOUD", "H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"]
    valid_levels = [
        34.0,
        39.0,
        41.0,
        43.0,
        44.0,
        45.0,
        48.0,
        51.0,
        53.0,
        56.0,
        63.0,
        68.0,
        71.0,
        72.0,
    ]
    def __init__(
            self,
            data_root_dir:str="../data",
            model_root_dir:str="../weights",
            data_surface_vars:list[str]=[],
            data_static_surface_vars:list[str]=[],
            data_levels:list[float]=[],
            download:bool=False,
            overwrite:bool=False,
            in_channels: int=160,
            input_size_time: int=2,
            in_channels_static: int=8,
            input_scalers_epsilon: float=0.0,
            static_input_scalers_epsilon: float=0.0,
            n_lats_px: int=360,
            n_lons_px: int=576,
            patch_size_px: tuple[int, int]=(2,2),
            mask_unit_size_px: tuple[int, int]=(30,32),
            mask_ratio_inputs: float=0.0,
            mask_ratio_targets: float=0.0,
            embed_dim: int=2560,
            n_blocks_encoder: int=12,
            n_blocks_decoder: int=2,
            mlp_multiplier: int=4,
            n_heads: int=16,
            dropout: float=0.0,
            drop_path: float=0.0,
            parameter_dropout: float=0.0,
            residual: str="climate",
            masking_mode: str="global",
            positional_encoding: str="fourier",
            encoder_shifting: bool = False,
            decoder_shifting: bool = False,
            checkpoint_encoder: list[int]=[],
            checkpoint_decoder: list[int]=[],
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
        self.local_data_root_dir = self._set_local_data_root_dir(
            data_root_dir=data_root_dir)
        self.local_model_root_dir = self._set_local_data_root_dir(
            data_root_dir=model_root_dir
        )
        self.download = download
        self.overwrite = overwrite

        #self._download_and_load_prithvi_wxc_scalers()
        
        super().__init__()
    
    def _set_local_data_root_dir(self, data_root_dir:str="../data")->LiteralString:
        if data_root_dir is None:
            data_root_dir = "../data"
        # make the directory if not exists
        os.makedirs(data_root_dir, exist_ok=True)
        return data_root_dir
    
    def _download_and_load_prithvi_wxc_scalers(self):
        """
            Download prithvi wxc scalers
            The WxC model takes as static parameters the mean and variance values of 
            the input variables and the variance values of the target difference, i.e., 
            the variance between climatology and instantaneous variables.

            Returns:
                input_scalers_mu: Tensor,
                input_scalers_sigma: Tensor,
                static_input_scalers_mu: Tensor,
                static_input_scalers_sigma: Tensor,
                output_scalers: Tensor,

            Raises:
                RepositoryNotFoundError — If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.
                RevisionNotFoundError — If the revision to download from cannot be found.
                EntryNotFoundError — If the file to download cannot be found.
                LocalEntryNotFoundError — If network is disabled or unavailable and file is not found in cache.
                EnvironmentError — If token=True but the token cannot be found.
                OSError — If ETag cannot be determined.
                ValueError — If some parameter value is invalid.
        """
        surf_in_scal_path = os.path.join(
            self.local_data_root_dir,
            "climatology/musigma_surface.nc")
        if self.download:
            if (self.overwrite or
                (not os.path.exists(surf_in_scal_path))):
                hf_hub_download(
                    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
                    filename=f"climatology/{os.path.basename(surf_in_scal_path)}",
                    local_dir=self.local_data_root_dir,
                )

        vert_in_scal_path = os.path.join(
            self.local_data_root_dir,
            "climatology/musigma_vertical.nc")
        if self.download:
            if (self.overwrite or
                (not os.path.exists(vert_in_scal_path))):
                hf_hub_download(
                    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
                    filename=f"climatology/{os.path.basename(vert_in_scal_path)}",
                    local_dir=self.local_data_root_dir,
            )

        surf_out_scal_path = os.path.join(
            self.local_data_root_dir,
            "climatology/anomaly_variance_surface.nc")
        if self.download:
            if (self.overwrite or
                (not os.path.exists(surf_out_scal_path))):
                hf_hub_download(
                    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
                    filename=f"climatology/{os.path.basename(surf_out_scal_path)}",
                    local_dir=self.local_data_root_dir,
                )

        vert_out_scal_path = os.path.join(
            self.local_data_root_dir,
            "climatology/anomaly_variance_vertical.nc")
        if self.download:
            if (self.overwrite or
                (not os.path.exists(vert_out_scal_path))):
                hf_hub_download(
                    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
                    filename=f"climatology/{os.path.basename(vert_out_scal_path)}",
                    local_dir=self.local_data_root_dir,
                )

        _input_scalers_mu, _input_scalers_sigma = input_scalers(
            self.valid_surface_vars,
            self.valid_vertical_vars,
            self.valid_levels,
            surf_in_scal_path,
            vert_in_scal_path,
        )

        _output_scalers = output_scalers(
            self.valid_surface_vars,
            self.valid_vertical_vars,
            self.valid_levels,
            surf_out_scal_path,
            vert_out_scal_path,
        )

        _static_input_scalers_mu, _static_input_scalers_sigma = static_input_scalers(
            surf_in_scal_path,
            self.valid_static_surface_vars,
        )
        self.in_mu = _input_scalers_mu
        self.in_sig = _input_scalers_sigma
        self.output_sig = _output_scalers
        self.static_mu = _static_input_scalers_mu
        self.static_sig = _static_input_scalers_sigma

    def configure_models(self):
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self._my_load_model_wxc()

    def _download_and_load_weights_for_prithvi_wxc(self):
        """
            Download and load weights for Prithvi WxC

            Returns:
                model uploaded with weitghts

            Raises:
                RepositoryNotFoundError — If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.
                RevisionNotFoundError — If the revision to download from cannot be found.
                EntryNotFoundError — If the file to download cannot be found.
                LocalEntryNotFoundError — If network is disabled or unavailable and file is not found in cache.
                EnvironmentError — If token=True but the token cannot be found.
                OSError — If ETag cannot be determined.
                ValueError — If some parameter value is invalid.
        """

        #weights_path = os.path.join(self.local_model_root_dir, "weights/prithvi.wxc.2300m.v1.pt")
        data_root_dir = "/mnt/f/ubf/srcexp/fd/terratorch/Prithvi-WxC/data"
        weights_path = os.path.join(data_root_dir, "weights/prithvi.wxc.2300m.v1.pt")
        if self.download:
            if (self.overwrite or
                    (not os.path.exists(weights_path))):
                hf_hub_download(
                    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
                    filename=os.path.basename(weights_path),
                    local_dir=self.local_weights_root_dir
                )

        state_dict = torch.load(weights_path, weights_only=False)
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        self.model.load_state_dict(state_dict, strict=True)


    def _my_load_model_wxc(self):
        """
            This method is not used. For test purpose.
        """
        data_root_dir = "/mnt/f/ubf/srcexp/fd/terratorch/Prithvi-WxC/data"
        with open(data_root_dir+"/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        model = PrithviWxC(
            in_channels=config["params"]["in_channels"],
            input_size_time=config["params"]["input_size_time"],
            in_channels_static=config["params"]["in_channels_static"],
            input_scalers_mu=self.in_mu,
            input_scalers_sigma=self.in_sig,
            input_scalers_epsilon=config["params"]["input_scalers_epsilon"],
            static_input_scalers_mu=self.static_mu,
            static_input_scalers_sigma=self.static_sig,
            static_input_scalers_epsilon=config["params"][
                "static_input_scalers_epsilon"
            ],
            output_scalers=self.output_sig**0.5,
            n_lats_px=config["params"]["n_lats_px"],
            n_lons_px=config["params"]["n_lons_px"],
            patch_size_px=config["params"]["patch_size_px"],
            mask_unit_size_px=config["params"]["mask_unit_size_px"],
            mask_ratio_inputs=self.hparams.get("mask_ratio_inputs", 0.0), #masking_ratio,
            mask_ratio_targets=0.0,
            embed_dim=config["params"]["embed_dim"],
            n_blocks_encoder=config["params"]["n_blocks_encoder"],
            n_blocks_decoder=config["params"]["n_blocks_decoder"],
            mlp_multiplier=self.hparams.get("mlp_multiplier", 4), #config["params"]["mlp_multiplier"],
            n_heads=config["params"]["n_heads"],
            dropout=config["params"]["dropout"],
            drop_path=config["params"]["drop_path"],
            parameter_dropout=config["params"]["parameter_dropout"],
            residual=self.hparams.get("residual", "climate"), #residual,
            masking_mode=self.hparams.get("masking_mode", "global"), #masking_mode,
            encoder_shifting=self.hparams.get("encoder_shifting", False), #encoder_shifting,
            decoder_shifting=self.hparams.get("decoder_shifting", False), #decoder_shifting,
            positional_encoding=self.hparams.get("positional_encoding", "fourier"), #positional_encoding,
            checkpoint_encoder=self.hparams.get("checkpoint_encoder", []),
            checkpoint_decoder=self.hparams.get("checkpoint_decoder", []),
        )
        print("model created")
        print(model)
