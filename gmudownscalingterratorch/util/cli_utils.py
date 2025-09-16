"""
    Utility functions
"""
import os
from lightning.pytorch.cli import ArgsType, LightningCLI
# Allow classes to be referenced using only the class name from torchgeo
import torchgeo.datamodules
import torchgeo.trainers
import gmudownscalingterratorch.tasks
import gmudownscalingterratorch.datamodules
import lightning as pl
from torchgeo.datamodules import BaseDataModule
from torchgeo.trainers import BaseTask

def gmu_build_lightning_cli(
        args: ArgsType = None,
        run: bool = True
) -> LightningCLI:
    """Command-line interface to TorchGeo."""
    # Taken from https://github.com/pangeo-data/cog-best-practices
    rasterio_best_practices = {
        'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
        'AWS_NO_SIGN_REQUEST': 'YES',
        'GDAL_MAX_RAW_BLOCK_CACHE_SIZE': '200000000',
        'GDAL_SWATH_SIZE': '200000000',
        'VSI_CURL_CACHE_SIZE': '200000000',
    }
    os.environ.update(rasterio_best_practices)
    
    return GmuDownscalingLightningCLI(
        model_class=BaseTask,
        datamodule_class=pl.LightningDataModule,
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={'overwrite': True},
        args=args,
    )

class GmuDownscalingLightningCLI(LightningCLI):
    def run(self):
        super().run()
    
    def instantiate_classes(self):
        return super().instantiate_classes()