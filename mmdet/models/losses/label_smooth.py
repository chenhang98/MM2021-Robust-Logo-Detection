import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss

# refer: https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch


@LOSSES.register_module()
class LabelSmoothingCrossEntropyLoss(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self,
                x,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        # assert weight is None
        assert reduction_override is None
        if weight is not None:
            weight = weight.float()

        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=label.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        loss = weight_reduce_loss(
            loss, weight=weight, reduction='mean', avg_factor=avg_factor)
        return loss
