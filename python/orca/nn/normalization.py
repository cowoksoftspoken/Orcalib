import orca
from .module import Module
from .parameter import Parameter
from orca.tensor import Tensor
from .activation import _scalar

class LayerNorm(Module):
    """
    Applies Layer Normalization over a mini-batch of inputs as described in the paper
    Layer Normalization.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5, device=None):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.weight = Parameter(Tensor.ones([1, normalized_shape], requires_grad=True, device=device))
        self.bias = Parameter(Tensor.zeros([1, normalized_shape], requires_grad=True, device=device))

    def forward(self, x: Tensor) -> Tensor:
        shape = list(x.shape)
        if shape[-1] != self.normalized_shape:
            raise ValueError(f"Expected last dim {self.normalized_shape}, got {shape[-1]}")
            
        reduced_shape = list(shape)
        reduced_shape[-1] = 1
        
        F = float(self.normalized_shape)
        
        sum_x = x.sum_to_shape(reduced_shape)
        mean = sum_x * (1.0 / F)
        
        diff = x - mean
        squared = diff * diff
        
        sum_sq = squared.sum_to_shape(reduced_shape)
        var = sum_sq * (1.0 / F)
        
        eps_tensor = _scalar(self.eps, x.device)
        var_eps = var + eps_tensor
        std = var_eps.sqrt()
        
        norm = diff / std
        return (norm * self.weight.tensor) + self.bias.tensor


class BatchNorm2d(Module):
    """
    Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension).
    This performs proper 4D batch normalization across the spatial and batch dimensions.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, device=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = Parameter(Tensor.ones([1, num_features, 1, 1], requires_grad=True, device=device))
        self.bias = Parameter(Tensor.zeros([1, num_features, 1, 1], requires_grad=True, device=device))
        
        # Buffers for running statistics
        self.register_buffer('running_mean', Tensor.zeros([1, num_features, 1, 1], requires_grad=False, device=device))
        self.register_buffer('running_var', Tensor.ones([1, num_features, 1, 1], requires_grad=False, device=device))

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        if len(shape) != 4:
            raise ValueError(f"BatchNorm2d expected 4D input, got {len(shape)}D")
            
        if self.training:
            # Compute mean and variance across N, H, W
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            num_elements = N * H * W
            
            reduced_shape = [1, C, 1, 1]
            
            sum_x = x.sum_to_shape(reduced_shape)
            mean = sum_x * (1.0 / num_elements)
            
            diff = x - mean
            squared = diff * diff
            sum_sq = squared.sum_to_shape(reduced_shape)
            
            # Bessel's correction for unbiased variance if N*H*W > 1
            unbiased_denom = max(1.0, float(num_elements - 1))
            var = sum_sq * (1.0 / num_elements)
            unbiased_var = sum_sq * (1.0 / unbiased_denom)
            
            # Update running stats (momentum)
            # Create disconnected tensors to prevent graph retention
            mean_vals = mean.to_list()
            var_vals = unbiased_var.to_list()
            
            new_running_mean = Tensor.from_list(mean_vals, reduced_shape, requires_grad=False, device=x.device)
            new_running_var = Tensor.from_list(var_vals, reduced_shape, requires_grad=False, device=x.device)
            
            # Update self._buffers directly
            old_rm = self._buffers['running_mean']
            old_rv = self._buffers['running_var']
            
            self._buffers['running_mean'] = (old_rm * (1.0 - self.momentum)) + (new_running_mean * self.momentum)
            self._buffers['running_var'] = (old_rv * (1.0 - self.momentum)) + (new_running_var * self.momentum)
            
            # Normalize and apply affine
            eps_tensor = _scalar(self.eps, x.device)
            norm = diff / (var + eps_tensor).sqrt()
            return (norm * self.weight.tensor) + self.bias.tensor
        else:
            # Evaluation mode uses running stats
            eps_tensor = _scalar(self.eps, x.device)
            norm = (x - self._buffers['running_mean']) / (self._buffers['running_var'] + eps_tensor).sqrt()
            return (norm * self.weight.tensor) + self.bias.tensor
