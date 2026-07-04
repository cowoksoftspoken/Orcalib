from .module import Module
from orca.tensor import Tensor

class MSELoss(Module):
    """
    Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input `x` and target `y`.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        squared = diff * diff
        
        if self.reduction == 'mean':
            return squared.mean()
        elif self.reduction == 'sum':
            return squared.sum()
        else:
            return squared

class CrossEntropyLoss(Module):
    """
    Computes the cross entropy loss between input logits and target.
    Target is expected to be one-hot encoded.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # pred: [Batch, Classes], target: [Batch, Classes] (one-hot)
        exp_pred = pred.exp()
        batch_size = pred.shape[0]
        
        # sum_exp: [Batch, 1]
        sum_exp = exp_pred.sum_to_shape([batch_size, 1])
        
        # log_sum_exp: [Batch, 1]
        log_sum_exp = sum_exp.log()
        
        # log_softmax: [Batch, Classes] (Broadcasting [B, C] - [B, 1])
        log_softmax = pred - log_sum_exp
        
        # loss_matrix: [Batch, Classes]
        # We now use true unary negation __neg__ instead of * -1.0
        loss_matrix = -(target * log_softmax)
        
        # Sum over classes: [Batch, 1]
        per_batch_loss = loss_matrix.sum_to_shape([batch_size, 1])
        
        if self.reduction == 'mean':
            return per_batch_loss.mean()
        elif self.reduction == 'sum':
            return per_batch_loss.sum()
        else:
            return per_batch_loss
