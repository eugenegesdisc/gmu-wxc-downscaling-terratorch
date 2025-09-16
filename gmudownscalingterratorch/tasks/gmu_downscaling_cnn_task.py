import logging
import torch as pt
import torch.nn as pt_nn
import torch.optim as optim
import torch.functional as pt_F
from torchgeo.trainers import BaseTask
from torch.optim.lr_scheduler import CosineAnnealingLR
from gmudownscalingterratorch.models import gmu_wxc_downscaling_model_factory
from gmudownscalingterratorch.datasets.downscaling.gmu_downscaling_nn_modules import(
    FiltedRegressionMeanLoss
)
from torchmetrics import MetricCollection
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score
)
from torchmetrics.image import StructuralSimilarityIndexMeasure

logger = logging.getLogger(__name__)

class GmuDowncalingCnnTask(BaseTask):

    # any keywords we add here between *args and **kwargs will be found in self.hparams
    def __init__(
        self,
        model_config=None,
        lr:float=0.01) -> None:
        super().__init__()  # pass args and kwargs to the parent class

    def configure_models(self):
        config = self.hparams.get("model_config",{})
        model_fac = gmu_wxc_downscaling_model_factory.GmuWxCDownscalingModelFactory()
        self.model = model_fac.build_model(
            downscaling_nn_model="downscaling_cnn",
            config=config
        )
        self.output_vars = config["data"]["output_vars"]
        self.data_prism_vars = config["data"]["data_prism_vars"]
        self.output_var_index = pt.tensor([ self.data_prism_vars.index(v) for v in self.output_vars])
        self.loss_fn = FiltedRegressionMeanLoss(
            mask=pt.Tensor([True]),
            pt_loss_func=pt_nn.MSELoss)

        
    def configure_optimizers(
        self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams['lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=-10)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.monitor},
        }

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        metrics = MetricCollection(
            {
                "MAE": MeanAbsoluteError(),
                "MSE": MeanSquaredError(),
                "R2": R2Score(),
                #"SSIM": StructuralSimilarityIndexMeasure(),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
    #    logger.info("metrics....")

    def _common_step(self, batch, batch_idx, stage):
        prediction = self.model(batch)
        
        #.output
        loss = self.loss_fn(prediction, batch)

        return loss

    def forward(self,x):
        print(x.keys())
        print("x[x].shape=", x["x"].shape)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        print("in training step....")
        print("args batch_idx=", batch_idx)
        prediction = self(batch)
        print('batch["hr_prism"].shape=', batch["hr_prism"].shape)
        target = batch["hr_prism"][0:1, self.output_var_index, :, :]
        if "hr_prism_mask" in batch:
            target_mask = batch["hr_prism_mask"][0:1, self.output_var_index, :, :]
            prediction = prediction * target_mask
            target = target * target_mask
            #self.loss_fn.mask = target_mask
        print("target.shape at train - step=", target.shape)
        loss = self.loss_fn(prediction,target)
        metrics = self.train_metrics(
            pt.flatten(pt.squeeze(prediction),start_dim=-2,end_dim=-1),
            pt.flatten(pt.squeeze(target),start_dim=-2, end_dim=-1))
        self.log_dict(metrics)
        #print("metrics-train=", metrics)

        print("loss at train =", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        print("batch.keys=", batch.keys())
        print("here.... in validation-step")
        print("batch_idx=", batch_idx)
        prediction = self(batch)
        print('batch["hr_prism"].shape=', batch["hr_prism"].shape)
        target = batch["hr_prism"][0:1, self.output_var_index, :, :]
        print("target.shape in validation=", target.shape)
        if "hr_prism_mask" in batch:
            target_mask = batch["hr_prism_mask"][0:1, self.output_var_index, :, :]
            print("target_mask.shape=", target_mask.shape)
            self.loss_fn.mask = target_mask
        loss = self.loss_fn(prediction,target)
        print("loss.shape=",loss.shape)
        print("prediction.shape=", prediction.shape)
        print("target.shape=", target.shape)

        #self.val_metrics.update(prediction,target)
        m_p = pt.flatten(pt.squeeze(prediction),start_dim=-2,end_dim=-1)
        m_t = pt.flatten(pt.squeeze(target),start_dim=-2, end_dim=-1)
        print("m_p.shape=", m_p.shape)
        print("m_t.shape=", m_t.shape)
        print("prediction.shape=", prediction.shape)
        print("target.shape=", target.shape)
        metrics = self.val_metrics.update(
            m_p,
            m_t)
        print("loss at validation =", loss)
        return loss
        #return self._common_step(batch, batch_idx, "validate")
    
    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)
        print("metrics-epoch-validation=", metrics)
        self.val_metrics.reset()


    def test_step(self, batch, batch_idx):
        print("here....")
        print("batch.keys=", batch.keys())
        print("here.... in test-step")
        print("batch_idx=", batch_idx)
        prediction = self(batch)
        print('batch["hr_prism"].shape=', batch["hr_prism"].shape)
        target = batch["hr_prism"][0:1, self.output_var_index, :, :]
        print("target.shape in test=", target.shape)
        if "hr_prism_mask" in batch:
            target_mask = batch["hr_prism_mask"][0:1, self.output_var_index, :, :]
            self.loss_fn.mask = target_mask
        loss = self.loss_fn(prediction,target)
        metrics = self.test_metrics(prediction,target)
        self.log_dict(metrics)
        #print("metrics-test=", metrics)
        return loss
    
    def predict_step(self, batch, batch_idx):
        print("here.....prediction")
        print("batch_idx=", batch_idx)
        predictions = self(batch)
        return predictions
