import orca
from .module import Module
from orca.tensor import Tensor
import random

class Dropout(Module):
    """
    During training, randomly zeroes some of the elements of the input tensor with probability p
    using samples from a Bernoulli distribution. Each channel will be zeroed out independently
    on every forward call.
    """
    def __init__(self, p=0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
            
        # Native Rust generation of the dropout mask
        mask = orca.Tensor.rand_dropout_mask(x.shape, self.p, device=x.device)
        return x * mask
