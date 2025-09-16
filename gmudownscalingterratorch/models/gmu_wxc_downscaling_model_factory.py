import torch
import numpy as np
from torch import nn
import os
import yaml

from PrithviWxC.model import PrithviWxC
from .gmu_conv_downscaling import ConvEncoderDecoder
from huggingface_hub import hf_hub_download
from PrithviWxC.dataloaders.merra2 import (
    input_scalers,
    output_scalers,
    static_input_scalers,
)


class GmuWxCDownscalingCnn(nn.Module):
    prithvi_surface_vars = [
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
    prithvi_static_surface_vars = [
        "FRACI",
        "FRLAND",
        "FROCEAN",
        "PHIS"]
    prithvi_vertical_vars = [
        "CLOUD",
        "H",
        "OMEGA",
        "PL",
        "QI",
        "QL",
        "QV",
        "T",
        "U",
        "V"]
    prithvi_levels = [
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

    # set default kernel size
    def __init__(
            self,
            config:dict):
        super(GmuWxCDownscalingCnn,self).__init__()
        config = self._set_default_kernel_size(config)
        self._set_embeddings(config)

        self.upscale = self._set_upscale(config)
        self.backbone = self._set_backbone(config)
        self.head = self._set_head(config)
    
    
    def _set_default_kernel_size(
            self,
            config:dict)->dict:
        if 'encoder_decoder_kernel_size_per_stage' not in config["model"]:
            config["model"]["encoder_decoder_kernel_size_per_stage"] = [
                [3]*len(
                    inner) for inner in config.get("model").get("encoder_decoder_scale_per_stage")]

        self.n_output_parameters = len(config["data"]["output_vars"])
        if config.get("model").get('loss_type', 'patch_rmse_loss')=='cross_entropy':
            if config.get("model").get('cross_entropy_bin_width_type', 'uniform') == 'uniform':
                self.n_output_parameters = config.get("model").get('cross_entropy_n_bins', 512)
            else:
                self.n_output_parameters = len(np.load(
                    config["model"]["cross_entropy_bin_boundaries_file"])) + 1

        return config
    
    def _set_upscale(self, config):
        self.upscale = ConvEncoderDecoder(
            in_channels=config["model"]["downscaling_embed_dim"],
            channels=config["model"]["encoder_decoder_conv_channels"],
            out_channels=config["model"]["embed_dim"],
            kernel_size=config["model"]["encoder_decoder_kernel_size_per_stage"][0],
            scale=config["model"]["encoder_decoder_scale_per_stage"][0],
            upsampling_mode=config["model"]["encoder_decoder_upsampling_mode"],
        )
        return self.upscale 
            
    
    def _set_embeddings(self, config):
        self.embedding = None
        self.embeding_static = None


    def _set_backbone(self, config):
        if config["model"]["backbone"] == "prithvi-wxc":
            return self._load_backbone_prithvi(
                model_config=config,
                repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
                weights_filename="prithvi.wxc.2300m.v1.pt",
                model_subdir="prithviwxc")
        elif config["model"]["backbone"] == "prithvi-wxc-rollout":
            return self._load_backbone_prithvi(
                model_config=config,
                repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M-rollout",
                weights_filename="prithvi.wxc.rollout.2300m.v1.pt",
                model_subdir="prithviwxcrollout")
        #return nn.Identity()
        def retrieve_time2(batch):
            """
                batch["x"]= [batch, time, parameter, lat, lon]
            """
            return batch["x"][0,:-1,:,:,:]
        return retrieve_time2

    def _load_scalers(
            self,
            model_root_dir:str="../models",
            repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
            )->dict:
        surf_in_scal_path =os.path.join(
            model_root_dir,
            "climatology",
            "musigma_surface.nc")
        if not os.path.exists(surf_in_scal_path):
            hf_hub_download(
                repo_id=repo_id,
                filename="climatology/musigma_surface.nc",
                local_dir=model_root_dir,
            )

        vert_in_scal_path = os.path.join(
            model_root_dir,
            "climatology",
            "musigma_vertical.nc")
        if not os.path.exists(vert_in_scal_path):
            hf_hub_download(
                repo_id=repo_id,
                filename="climatology/musigma_vertical.nc",
                local_dir=model_root_dir,
            )

        surf_out_scal_path = os.path.join(
            model_root_dir,
            "climatology",
            "anomaly_variance_surface.nc")
        if not os.path.exists(surf_out_scal_path):
            hf_hub_download(
                repo_id=repo_id,
                filename="climatology/anomaly_variance_surface.nc",
                local_dir=model_root_dir,
            )

        vert_out_scal_path = os.path.join(
            model_root_dir,
            "climatology",
            "anomaly_variance_vertical.nc")
        if not os.path.exists(vert_out_scal_path):
            hf_hub_download(
                repo_id=repo_id,
                filename="climatology/anomaly_variance_vertical.nc",
                local_dir=model_root_dir,
            )

        in_mu, in_sig = input_scalers(
            self.prithvi_surface_vars,
            self.prithvi_vertical_vars,
            self.prithvi_levels,
            surf_in_scal_path,
            vert_in_scal_path,
        )

        output_sig = output_scalers(
            self.prithvi_surface_vars,
            self.prithvi_vertical_vars,
            self.prithvi_levels,
            surf_out_scal_path,
            vert_out_scal_path,
        )

        static_mu, static_sig = static_input_scalers(
            surf_in_scal_path,
            self.prithvi_static_surface_vars,
        )
        ret_dict = {
            "in_mu": in_mu,
            "in_sig": in_sig,
            "output_sig": output_sig,
            "static_mu": static_mu,
            "static_sig": static_sig
        }
        return ret_dict


    def _load_backbone_prithvi(
            self,
            model_config:dict,
            model_subdir:str,
            repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
            weights_filename="prithvi.wxc.2300m.v1.pt",
            )->nn.Module:
        #ret_model = nn.Identity()
        model_root_dir = os.path.join(
            model_config["model"]["backbone_model_root_dir"],
            model_subdir)
        # make the directory if it does not exist
        os.makedirs(model_root_dir, exist_ok=True)
        model_default_config_file = os.path.join(
            model_root_dir, "config.yaml")
        if not os.path.exists(model_default_config_file):
            hf_hub_download(
                repo_id=repo_id,
                filename="config.yaml",
                local_dir=model_root_dir,
            )
        with open(model_default_config_file, "r") as f:
            config = yaml.safe_load(f)

        scalers = self._load_scalers(
            model_root_dir=model_root_dir,
            repo_id=repo_id)
        positional_encoding = config["params"].get("positional_encoding","fourier")
        residual = config["params"].get("residual","climate")
        masking_mode = config["params"].get("masking_mode","local")
        encoder_shifting = config["params"].get("encoder_shifting",True)
        decoder_shifting = config["params"].get("decoder_shifting",True)
        masking_ratio = config["params"].get("masking_ratio",0.0)

        ret_model = PrithviWxC(
            in_channels=config["params"]["in_channels"],
            input_size_time=config["params"]["input_size_time"],
            in_channels_static=config["params"]["in_channels_static"],
            input_scalers_mu=scalers["in_mu"],
            input_scalers_sigma=scalers["in_sig"],
            input_scalers_epsilon=config["params"]["input_scalers_epsilon"],
            static_input_scalers_mu=scalers["static_mu"],
            static_input_scalers_sigma=scalers["static_sig"],
            static_input_scalers_epsilon=config["params"][
                "static_input_scalers_epsilon"
            ],
            output_scalers=scalers["output_sig"]**0.5,
            n_lats_px=config["params"]["n_lats_px"],
            n_lons_px=config["params"]["n_lons_px"],
            patch_size_px=config["params"]["patch_size_px"],
            mask_unit_size_px=config["params"]["mask_unit_size_px"],
            mask_ratio_inputs=masking_ratio,
            mask_ratio_targets=0.0,
            embed_dim=config["params"]["embed_dim"],
            n_blocks_encoder=config["params"]["n_blocks_encoder"],
            n_blocks_decoder=config["params"]["n_blocks_decoder"],
            mlp_multiplier=config["params"]["mlp_multiplier"],
            n_heads=config["params"]["n_heads"],
            dropout=config["params"]["dropout"],
            drop_path=config["params"]["drop_path"],
            parameter_dropout=config["params"]["parameter_dropout"],
            residual=residual,
            masking_mode=masking_mode,
            encoder_shifting=encoder_shifting,
            decoder_shifting=decoder_shifting,
            positional_encoding=positional_encoding,
            checkpoint_encoder=[],
            checkpoint_decoder=[],
        )

        # Model init and load weights
        model_pretrained_weights_file = os.path.join(
            model_root_dir, weights_filename)
        if not os.path.exists(model_pretrained_weights_file):
            hf_hub_download(
                repo_id=repo_id,
                filename=weights_filename,
                local_dir=model_root_dir,
            )
        
        
        state_dict = torch.load(
            model_pretrained_weights_file, weights_only=False)
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        ret_model.load_state_dict(state_dict, strict=True)
        

        """
        #-----------DESKTOP
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        ret_model.load_state_dict(
            torch.load(
                "/mnt/f/ubf/srcexp/fd/terratorch/Prithvi-WxC/data/weights/prithvi.wxc.2300m.v1.bin",
                weights_only=False,
                map_location=device))
        #------------
        """

        print("backbone LOADED")
        freeze_backbone = model_config["model"].get("freeze_backbone", True)
        print("freeze_back=", freeze_backbone)
        if freeze_backbone:
            for param in ret_model.parameters():
                param.requires_grad = False
            print("backbone weights frozen")
        return ret_model
    def _set_head(self, config):
        self.head = ConvEncoderDecoder(
                in_channels=config["model"]["embed_dim"],
                channels=config["model"]["encoder_decoder_conv_channels"],
                out_channels=self.n_output_parameters,
                kernel_size=config["model"]["encoder_decoder_kernel_size_per_stage"][1],
                scale=config["model"]["encoder_decoder_scale_per_stage"][1],
                upsampling_mode=config["model"]["encoder_decoder_upsampling_mode"],
        )
        return self.head

    def forward(self, batch: dict[str, torch.tensor]):
        """
        Args:
            batch: Dictionary containing the keys 'x', 'y', and 'static'.
                The associated torch tensors have the following shapes:
                x: Tensor of shape [batch, time x parameter, lat, lon]
                y: Tensor of shape [batch, parameter, lat, lon]
                static: Tensor of shape [batch, channel_static, lat, lon]
                climate: Optional tensor of shape [batch, parameter, lat, lon]
                hr_static: Tensor of shape [batch, channel, channel_static, hr_lat, hr_lon]
                prism: tensor of shape [batch, hr_parameter, hr_lat, hr_lon]
        Returns:
            Tensor of shape [batch, hr_parameter, hr_lat, hr_lon].

        """
        print("Entering backbone...")
        x = batch["x"]
        print("before backbone: x.shape=", x.shape)
        print("channels=",x.shape[2])
        #batch["x"] = x
        print("y.shape=", batch["y"].shape)
        print("climate.shape=", batch["climate"].shape)
        print("static.shape=", batch["static"].shape)
        x_bb_out = self.backbone(batch)
        print("x_bb_out.shape=", x_bb_out.shape)
        # x_bb_out should be [batch, param, lat, lon]
        # Expand dimensions of x_bb_out to [batch, time, param, lat, lon]
        ex_bb_out = x_bb_out.unsqueeze(1)
        print("ex_bb_out.shape=", ex_bb_out.shape)
        # x should be [batch, time, param, lat, lon]
        x0 = x
        x = torch.cat([x0, ex_bb_out],dim=1)
        print("x.shape=", x.shape)
        x1 = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))
        print("x1.shape=", x1.shape)
        x_out = self.head(x1)
        print("x_out.shape=", x_out.shape)
        print("Returning from backbone...")
        return x_out

class GmuWxCDownscalingSrcnn(nn.Module):
    # set default kernel size
    def __init__(
            self,
            config:dict):
        super(GmuWxCDownscalingCnn,self).__init__()
        config = self._set_default_kernel_size(config)
        self._set_embeddings(config)

        self.upscale = self._set_upscale(config)
        self.backbone = self._set_backbone(config)
        self.head = self._set_head(config)
    
    
    def _set_default_kernel_size(
            self,
            config:dict)->dict:
        if 'encoder_decoder_kernel_size_per_stage' not in config["model"]:
            config["model"]["encoder_decoder_kernel_size_per_stage"] = [
                [3]*len(
                    inner) for inner in config.get("model").get("encoder_decoder_scale_per_stage")]

        self.n_output_parameters = len(config["data"]["output_vars"])
        if config.get("model").get('loss_type', 'patch_rmse_loss')=='cross_entropy':
            if config.get("model").get('cross_entropy_bin_width_type', 'uniform') == 'uniform':
                self.n_output_parameters = config.get("model").get('cross_entropy_n_bins', 512)
            else:
                self.n_output_parameters = len(np.load(
                    config["model"]["cross_entropy_bin_boundaries_file"])) + 1

        return config
    
    def _set_upscale(self, config):
        self.upscale = ConvEncoderDecoder(
            in_channels=config["model"]["downscaling_embed_dim"],
            channels=config["model"]["encoder_decoder_conv_channels"],
            out_channels=config["model"]["embed_dim"],
            kernel_size=config["model"]["encoder_decoder_kernel_size_per_stage"][0],
            scale=config["model"]["encoder_decoder_scale_per_stage"][0],
            upsampling_mode=config["model"]["encoder_decoder_upsampling_mode"],
        )
        return self.upscale 
            
    
    def _set_embeddings(self, config):
        self.embedding = None
        self.embeding_static = None


    def _set_backbone(self, config):
        return nn.Identity()
    
    def _set_head(self, config):
        self.head = ConvEncoderDecoder(
                in_channels=config["model"]["embed_dim"],
                channels=config["model"]["encoder_decoder_conv_channels"],
                out_channels=self.n_output_parameters,
                kernel_size=config["model"]["encoder_decoder_kernel_size_per_stage"][1],
                scale=config["model"]["encoder_decoder_scale_per_stage"][1],
                upsampling_mode=config["model"]["encoder_decoder_upsampling_mode"],
        )
        return self.head

    def forward(self, batch: dict[str, torch.tensor]):
        """
        Args:
            batch: Dictionary containing the keys 'x', 'y', and 'static'.
                The associated torch tensors have the following shapes:
                x: Tensor of shape [batch, time x parameter, lat, lon]
                y: Tensor of shape [batch, parameter, lat, lon]
                static: Tensor of shape [batch, channel_static, lat, lon]
                climate: Optional tensor of shape [batch, parameter, lat, lon]
                hr_static: Tensor of shape [batch, channel, channel_static, hr_lat, hr_lon]
                prism: tensor of shape [batch, hr_parameter, hr_lat, hr_lon]
        Returns:
            Tensor of shape [batch, hr_parameter, hr_lat, hr_lon].

        """
        x = batch["x"][0]
        x_bb_out = self.backbone(x)
        print("x_bb_out.shape=", x_bb_out.shape)
        x0 = x
        x = x_bb_out
        print("x.shape=", x.shape)
        x1 = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))
        print("x1.shape=", x1.shape)
        x_out = self.head(x1)
        print("x_out.shape=", x_out.shape)
        return x_out

class GmuWxCDownscalingUnet(nn.Module):
    # set default kernel size
    def __init__(
            self,
            config:dict):
        super(GmuWxCDownscalingCnn,self).__init__()
        config = self._set_default_kernel_size(config)
        self._set_embeddings(config)

        self.upscale = self._set_upscale(config)
        self.backbone = self._set_backbone(config)
        self.head = self._set_head(config)
    
    
    def _set_default_kernel_size(
            self,
            config:dict)->dict:
        if 'encoder_decoder_kernel_size_per_stage' not in config["model"]:
            config["model"]["encoder_decoder_kernel_size_per_stage"] = [
                [3]*len(
                    inner) for inner in config.get("model").get("encoder_decoder_scale_per_stage")]

        self.n_output_parameters = len(config["data"]["output_vars"])
        if config.get("model").get('loss_type', 'patch_rmse_loss')=='cross_entropy':
            if config.get("model").get('cross_entropy_bin_width_type', 'uniform') == 'uniform':
                self.n_output_parameters = config.get("model").get('cross_entropy_n_bins', 512)
            else:
                self.n_output_parameters = len(np.load(
                    config["model"]["cross_entropy_bin_boundaries_file"])) + 1

        return config
    
    def _set_upscale(self, config):
        self.upscale = ConvEncoderDecoder(
            in_channels=config["model"]["downscaling_embed_dim"],
            channels=config["model"]["encoder_decoder_conv_channels"],
            out_channels=config["model"]["embed_dim"],
            kernel_size=config["model"]["encoder_decoder_kernel_size_per_stage"][0],
            scale=config["model"]["encoder_decoder_scale_per_stage"][0],
            upsampling_mode=config["model"]["encoder_decoder_upsampling_mode"],
        )
        return self.upscale 
            
    
    def _set_embeddings(self, config):
        self.embedding = None
        self.embeding_static = None


    def _set_backbone(self, config):
        return nn.Identity()
    
    def _set_head(self, config):
        self.head = ConvEncoderDecoder(
                in_channels=config["model"]["embed_dim"],
                channels=config["model"]["encoder_decoder_conv_channels"],
                out_channels=self.n_output_parameters,
                kernel_size=config["model"]["encoder_decoder_kernel_size_per_stage"][1],
                scale=config["model"]["encoder_decoder_scale_per_stage"][1],
                upsampling_mode=config["model"]["encoder_decoder_upsampling_mode"],
        )
        return self.head

    def forward(self, batch: dict[str, torch.tensor]):
        """
        Args:
            batch: Dictionary containing the keys 'x', 'y', and 'static'.
                The associated torch tensors have the following shapes:
                x: Tensor of shape [batch, time x parameter, lat, lon]
                y: Tensor of shape [batch, parameter, lat, lon]
                static: Tensor of shape [batch, channel_static, lat, lon]
                climate: Optional tensor of shape [batch, parameter, lat, lon]
                hr_static: Tensor of shape [batch, channel, channel_static, hr_lat, hr_lon]
                prism: tensor of shape [batch, hr_parameter, hr_lat, hr_lon]
        Returns:
            Tensor of shape [batch, hr_parameter, hr_lat, hr_lon].

        """
        x = batch["x"][0]
        x_bb_out = self.backbone(x)
        print("x_bb_out.shape=", x_bb_out.shape)
        x0 = x
        x = x_bb_out
        print("x.shape=", x.shape)
        x1 = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))
        print("x1.shape=", x1.shape)
        x_out = self.head(x1)
        print("x_out.shape=", x_out.shape)
        return x_out


class GmuWxCDownscalingModelFactory:
    """
        Downscaling Model factory.
        1. Create CNN, SRCNN, and Unet downscaling models without backbone
        2. Create CNN, SRCNN, and Unet downscaling models with backbone="prithviwxc"
        3. Create CNN, SRCNN, and Unet downscaling models with backbone="prithviwxcdownscaling"
        4. Create CNN, SRCNN, and Unet downscaling models with backbone="prithviwxcecccdownscaling"
    """
    def build_model(
            self,
            downscaling_nn_model:str,
            config:dict
            ):
        """
            Create downscaling model.

            Args:
                downscaling_nn_model: The downscaling framework - choiseces are
                    downscaling_cnn, downscaling_srcnn, and downscaling_unet
                pre_adapter: embedding or scaling adapter. Current choises are "mlp" and "conv2d".
                backbone: pre-trained foundation model. Current choices are "prithviwxc",
                    "prithviwxcdownscaling", and "prithviwxcecccdownscaling".
                post_adapter: head adapter. Current choises are "mlp" and "conv2d".
        """
        if downscaling_nn_model == "downscaling_cnn":
            return self._build_downscaling_cnn(config=config)
        elif downscaling_nn_model == "downscaling_srcnn":
            return self._build_downsclaing_srcnn()
        elif downscaling_nn_model == "downscaling_unet":
            return self._build_downsclaing_unet()
        else:
             raise NotImplementedError(
                 "Model: '%s' not recognized or not implemented",
                 downscaling_nn_model)

    def _build_downscaling_cnn(
            self,
            config
            )->nn.Module:
        return GmuWxCDownscalingCnn(config=config)

    def _build_downsclaing_srcnn(
            self,
            )->nn.Module:
        return GmuWxCDownscalingSrcnn()

    def _build_downsclaing_unet(
            self,
            )->nn.Module:
        return GmuWxCDownscalingUnet()
    