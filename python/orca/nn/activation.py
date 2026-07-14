from .module import Module
from orca.tensor import Tensor
import orca

class ReLU(Module):
    """
    Applies the rectified linear unit function element-wise.
    
    Formula: `ReLU(x) = max(0, x)`
    """
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for ReLU.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output tensor with ReLU applied.
        """
        return x.relu()

class Sigmoid(Module):
    """
    Applies the sigmoid function element-wise.
    
    Formula: `Sigmoid(x) = 1 / (1 + exp(-x))`
    """
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for Sigmoid.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output tensor with Sigmoid applied.
        """
        return x.sigmoid()

def _scalar(val: float, device) -> Tensor:
    """Helper function to create a scalar tensor."""
    return Tensor.from_list([val], shape=[1], device=device)

class Tanh(Module):
    """
    Applies the Hyperbolic Tangent (Tanh) function element-wise.
    
    Formula: `Tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)`
    """
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for Tanh.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output tensor with Tanh applied.
        """
        two_x = x * 2.0
        # Numerically stable clamping to prevent exp() overflow to infinity, which causes NaN (inf/inf)
        eighty = _scalar(80.0, x.device)
        neg_eighty = _scalar(-80.0, x.device)
        
        clamped = eighty - (eighty - two_x).relu()
        clamped = neg_eighty + (clamped - neg_eighty).relu()
        
        exp_2x = clamped.exp()
        one = _scalar(1.0, x.device)
        return (exp_2x - one) / (exp_2x + one)

class GELU(Module):
    """
    Applies the Gaussian Error Linear Unit (GELU) function.
    
    This uses the approximation: `0.5 * x * (1 + Tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
    """
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for GELU.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output tensor with GELU applied.
        """
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
    
    Rescales the elements to the range [0, 1] such that the elements along the specified
    dimension sum to 1.
    
    Args:
        dim (int): A dimension along which Softmax will be computed (so every slice 
            along dim will sum to 1). Default: -1.
    """
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for Softmax.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output tensor with Softmax applied.
        """
        shape = list(x.shape)
        # handle negative dims
        dim = self.dim
        if dim < 0:
            dim += len(shape)
            
        shape[dim] = 1
        
        # Numerically stable softmax: shift by max
        max_x = x.max_to_shape(shape).expand(list(x.shape))
        shifted_x = x - max_x
        exp_x = shifted_x.exp()
        sum_x = exp_x.sum_to_shape(shape).expand(list(x.shape))
        
        return exp_x / sum_x
