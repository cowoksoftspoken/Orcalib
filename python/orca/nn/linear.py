import math
from orca import Tensor
from .module import Module
from .parameter import Parameter

class Linear(Module):
    """
    Applies a linear transformation to the incoming data: y = x @ W + b.
    W is shaped (in_features, out_features).
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        
        # Kaiming-like initialization scaling (Uniform)
        bound = math.sqrt(1.0 / in_features) if in_features > 0 else 0.0
        
        w_tensor = Tensor.rand_uniform([in_features, out_features], -bound, bound, requires_grad=True)
        b_tensor = Tensor.rand_uniform([1, out_features], -bound, bound, requires_grad=True)
        
        self.weight = Parameter(w_tensor)
        self.bias = Parameter(b_tensor)

    def forward(self, x: Tensor) -> Tensor:
        # x is (batch, in_features)
        # weight is (in_features, out_features)
        # output is (batch, out_features)
        return (x @ self.weight.tensor) + self.bias.tensor
