from .module import Module
from orca.tensor import Tensor

class MSELoss(Module):
    """
    Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input `pred` and target `target`.
    
    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum
            of the output will be divided by the number of elements, 'sum': the output will
            be summed. Default: 'mean'.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass for MSELoss.
        
        Args:
            pred (Tensor): Predictions tensor.
            target (Tensor): Target/labels tensor of the same shape as pred.
            
        Returns:
            Tensor: Loss tensor. Scalar if reduction is 'mean' or 'sum', otherwise has the same shape as inputs.
        """
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
    
    This criterion combines `LogSoftmax` and `NLLLoss` in one single class.
    
    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass for CrossEntropyLoss.
        
        Args:
            pred (Tensor): Logits (unnormalized scores) tensor of shape `(Batch, Classes)`.
            target (Tensor): Target tensor of shape `(Batch, Classes)`, one-hot encoded.
            
        Returns:
            Tensor: Loss tensor. Scalar if reduction is 'mean' or 'sum'.
        """
        # pred: [Batch, Classes], target: [Batch, Classes] (one-hot)
        batch_size = pred.shape[0]
        
        # LogSumExp trick for numerical stability
        max_pred_base = pred.max_to_shape([batch_size, 1])
        max_pred = max_pred_base.expand(list(pred.shape))
        shifted_pred = pred - max_pred
        
        exp_pred = shifted_pred.exp()
        sum_exp_base = exp_pred.sum_to_shape([batch_size, 1])
        sum_exp = sum_exp_base.expand(list(pred.shape))
        
        # log_sum_exp needs to be expanded too
        log_sum_exp = max_pred + sum_exp.log()
        
        # log_softmax: [Batch, Classes]
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
