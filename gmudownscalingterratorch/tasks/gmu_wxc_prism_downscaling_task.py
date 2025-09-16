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
logger = logging.getLogger(__name__)


class GmuWxcPrismDownscalingTask(BaseTask):
    """
        Task to downscaling with data (Merra2 -- PRISM climate data)
    """
    def __init__(
            self,
            ):
        super().__init__()
    
    def configure_optimizers(self):
        return super().configure_optimizers()
    
    def configure_model(self):
        return super().configure_model()
    
    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)
    
    def test_step(self, *args, **kwargs):
        return super().test_step(*args, **kwargs)
    
    def predict_step(self, *args, **kwargs):
        return super().predict_step(*args, **kwargs)
