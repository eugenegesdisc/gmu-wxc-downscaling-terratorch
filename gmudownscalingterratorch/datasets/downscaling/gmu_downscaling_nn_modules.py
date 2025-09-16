import torch
import torch.nn as nn

class FiltedRegressionMeanLoss(nn.Module):
    def __init__(self, mask, pt_loss_func):
        """
        
        Args:
            mask: a mask where True to be filetered.
            pt_loss_func: examples are MSELoss, L1Loss, SmoothL1Loss
        """
        super(FiltedRegressionMeanLoss, self).__init__()
        self._mask = mask
        self.pt_loss_fn = pt_loss_func(reduction="none")

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask

    def forward(self, y_pred, y_true):
        elem_wise_loss = self.pt_loss_fn(y_pred, y_true)
        masked_loss = elem_wise_loss * self._mask.float().to(elem_wise_loss.device)

        agg_loss = masked_loss.mean()
        return agg_loss
