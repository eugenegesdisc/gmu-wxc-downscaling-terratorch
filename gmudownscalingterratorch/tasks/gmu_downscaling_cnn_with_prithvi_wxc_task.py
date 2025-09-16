import logging
import torch.optim as optim
from torchgeo.trainers import BaseTask
from gmudownscalingterratorch.models import gmu_wxc_downscaling_model_factory

logger = logging.getLogger(__name__)

class GmuDowncalingCnnWtihPrivthviWxcTask(BaseTask):
    # any keywords we add here between *args and **kwargs will be found in self.hparams
    def __init__(
        self,
        model_config=None) -> None:
        super().__init__()  # pass args and kwargs to the parent class

    def configure_models(self):
        config = self.hparams.get("model_config",{})
        #config["model"] = self.hparams.get("model_args",{})
        #config["data"] = self.hparams.get("model_data",{})
        model_fac = gmu_wxc_downscaling_model_factory.GmuWxCDownscalingModelFactory()
        self.model = model_fac.build_model(
            downscaling_nn_model="downscaling_cnn",
            config=config
        )

        
    def configure_optimizers(
        self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams['lr'])
        #scheduler = CosineAnnealingLR(optimizer, T_max=tmax, eta_min=eta_min)
        #    'lr_scheduler': {'scheduler': scheduler, 'monitor': self.monitor},
        return {
            'optimizer': optimizer,
        }

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        logger.info("metrics....")

    def training_step(self, *args, **kwargs):
        return super().training_step(*args, **kwargs)
    
    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)
    
    def test_step(self, *args, **kwargs):
        return super().test_step(*args, **kwargs)
    
    def predict_step(self, *args, **kwargs):
        return super().predict_step(*args, **kwargs)
