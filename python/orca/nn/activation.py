from .module import Module
from orca.tensor import Tensor

class ReLU(Module):
    """
    Applies the rectified linear unit function element-wise.
    """
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class Sigmoid(Module):
    """
    Applies the sigmoid function element-wise.
    """
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

def _scalar(val: float, device) -> Tensor:
    return Tensor.from_list([val], shape=[1], device=device)

class Tanh(Module):
    """
    Applies the Hyperbolic Tangent (Tanh) function element-wise.
    """
    def forward(self, x: Tensor) -> Tensor:
        two_x = x * 2.0
        exp_2x = two_x.exp()
        one = _scalar(1.0, x.device)
        return (exp_2x - one) / (exp_2x + one)

class GELU(Module):
    """
    Applies the Gaussian Error Linear Units function.
    Using the approximation: 0.5 * x * (1 + Tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    def forward(self, x: Tensor) -> Tensor:
        # math.sqrt(2/math.pi) is approx 0.7978845608
        sqrt_2_pi = 0.7978845608
        
        # x^3
        x_cubed = (x * x) * x
        
        # 0.044715 * x^3
        term1 = x_cubed * 0.044715
        
        # x + 0.044715 * x^3
        term2 = x + term1
        
        # sqrt(2/pi) * (x + 0.044715 * x^3)
        inner = term2 * sqrt_2_pi
        
        # tanh(inner) via the Tanh module to reuse logic
        tanh_module = Tanh()
        tanh_res = tanh_module.forward(inner)
        
        # 1 + tanh(...)
        one = _scalar(1.0, x.device)
        one_plus_tanh = tanh_res + one
        
        # 0.5 * x * (1 + tanh)
        return (x * 0.5) * one_plus_tanh

class Softmax(Module):
    """
    Applies the Softmax function to an n-dimensional input Tensor.
    """
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        exp_x = x.exp()
        
        shape = list(x.shape)
        # handle negative dims
        dim = self.dim
        if dim < 0:
            dim += len(shape)
            
        shape[dim] = 1
        sum_x = exp_x.sum_to_shape(shape)
        
        return exp_x / sum_x

