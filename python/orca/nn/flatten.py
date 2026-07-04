from .module import Module
from orca.tensor import Tensor

class Flatten(Module):
    """
    Flattens all dimensions except the first (batch) dimension.
    """
    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        if len(shape) < 2:
            return x
        batch_size = shape[0]
        rest = 1
        for s in shape[1:]:
            rest *= s
        return x.reshape([batch_size, rest])
