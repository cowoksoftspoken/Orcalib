import math
from orca import Tensor
from .module import Module
from .parameter import Parameter

class Linear(Module):
    """
    Applies a linear transformation to the incoming data: y = x @ W + b.
    W is shaped (in_features, out_features).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Kaiming He Initialization (scaled for ReLU typically, here just a normal distribution based on fan_in)
        stdv = 1.0 / (in_features ** 0.5)
        self.weight = Parameter(Tensor.randn([in_features, out_features], mean=0.0, std=stdv, requires_grad=True))
        if bias:
            self.bias = Parameter(Tensor.zeros([1, out_features], requires_grad=True))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        rank = len(original_shape)
        
        if rank > 2:
            # Flatten to 2D: [batch * seq_len * ..., in_features]
            flat_size = 1
            for d in original_shape[:-1]:
                flat_size *= d
            flat_shape = [flat_size, self.in_features]
            x_flat = x.reshape(flat_shape)
        else:
            x_flat = x
            
        out = x_flat @ self.weight.tensor
        
        if self.bias is not None:
            # expand bias to match out shape and add
            b = self.bias.tensor.expand(out.shape)
            out = out + b
            
        if rank > 2:
            # Reshape back to [batch, seq_len, ..., out_features]
            new_shape = list(original_shape[:-1]) + [self.out_features]
            out = out.reshape(new_shape)
            
        return out
