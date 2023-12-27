import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/d45f8908ab2f0246ba204c702a6161c9eb25f902/loss.py#L4
    """
    def __init__(self, smooth: float = 1.0, reduction='sum'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        assert inputs.size() == targets.size()
        for i in range(len(targets)):
            y_pred_ = inputs[i].contiguous().view(-1)
            y_true_ = targets[i].contiguous().view(-1)
            intersection = (y_pred_ * y_true_).sum()
            dsc = (2. * intersection + self.smooth) / (
                y_pred_.sum() + y_true_.sum() + self.smooth
            )
            dsc_loss = 1. - dsc
            if i == 0:
                loss = dsc_loss
            else:
                loss += dsc_loss
        if self.reduction == 'mean':
            loss /= len(y_true)
            
        return loss
    
