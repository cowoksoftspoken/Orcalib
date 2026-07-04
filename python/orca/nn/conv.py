import math
from orca.nn.module import Module
from orca.nn.parameter import Parameter
from orca.tensor import Tensor
import orca

class Conv2d(Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int | tuple[int, int], 
        stride: int = 1, 
        padding: int = 0, 
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        device=None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Kaiming uniform initialization
        # fan_in is calculated over the grouped input channels
        in_channels_per_group = self.in_channels // self.groups
        fan_in = in_channels_per_group * self.kernel_size[0] * self.kernel_size[1]
        
        # Calculate bounds: typical Kaiming Uniform with a = sqrt(5) gives bound = sqrt(3) * std
        # std = sqrt(2 / ( (1 + a^2) * fan_in )) => std = sqrt(2 / (6 * fan_in)) = sqrt(1 / (3 * fan_in))
        # bound = sqrt(3) * sqrt(1 / (3 * fan_in)) = sqrt(1 / fan_in)
        bound = math.sqrt(1.0 / fan_in) if fan_in > 0 else 0.0
        
        weight_tensor = Tensor.rand_uniform(
            [out_channels, in_channels_per_group, self.kernel_size[0], self.kernel_size[1]], 
            -bound, bound, 
            device=device, 
            requires_grad=True
        )
        self.weight = Parameter(weight_tensor)
        
        if bias:
            bias_tensor = Tensor.rand_uniform([1, out_channels], -bound, bound, device=device, requires_grad=True)
            self.bias = Parameter(bias_tensor)
        else:
            self.bias = None
            
    def forward(self, x: Tensor) -> Tensor:
        bias_tensor = self.bias.tensor if self.bias is not None else None
        return x.conv2d(self.weight.tensor, bias_tensor, self.padding, self.stride, self.dilation, self.groups)
