import math
from typing import Optional, Union, Tuple
from orca.nn.module import Module
from orca.nn.parameter import Parameter
from orca.tensor import Tensor
import orca

class Conv2d(Module):
    """
    Applies a 2D convolution over an input signal composed of several input planes.
    
    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolving kernel.
        stride (int, optional): Stride of the convolution. Default: 1.
        padding (int, optional): Zero-padding added to both sides of the input. Default: 0.
        dilation (int, optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True.
        device (Optional[orca.Device], optional): Device to initialize the parameters on.
        
    Attributes:
        weight (Parameter): The learnable weights of the module of shape `(out_channels, in_channels / groups, kernel_size[0], kernel_size[1])`.
        bias (Optional[Parameter]): The learnable bias of the module of shape `(1, out_channels)`.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: Union[int, Tuple[int, int]], 
        stride: int = 1, 
        padding: int = 0, 
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        device: Optional['orca.Device'] = None
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
        
        in_channels_per_group = self.in_channels // self.groups
        fan_in = in_channels_per_group * self.kernel_size[0] * self.kernel_size[1]
        
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
        """
        Forward pass of the Conv2d layer.
        
        Args:
            x (Tensor): Input tensor of shape `(batch, in_channels, in_height, in_width)`.
            
        Returns:
            Tensor: Output tensor of shape `(batch, out_channels, out_height, out_width)`.
        """
        bias_tensor = self.bias.tensor if self.bias is not None else None
        return x.conv2d(self.weight.tensor, bias_tensor, self.padding, self.stride, self.dilation, self.groups)
