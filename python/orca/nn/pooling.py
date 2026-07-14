from typing import Union, Tuple
from orca.nn.module import Module
from orca.tensor import Tensor
import orca
import numpy as np

class MaxPool2d(Module):
    """
    Applies a 2D max pooling over an input signal composed of several input planes.
    """
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]] = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        # Convert to numpy for stable sliding window max calculation
        x_np = np.array(x.to_list(), dtype=np.float32).reshape(x.shape)
        
        # Apply padding if any
        if self.padding > 0:
            x_np = np.pad(
                x_np, 
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                mode='constant'
            )
            
        b, c, h, w = x_np.shape
        kh, kw = (self.kernel_size, self.kernel_size) if isinstance(self.kernel_size, int) else self.kernel_size
        sh, sw = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
        
        oh = (h - kh) // sh + 1
        ow = (w - kw) // sw + 1
        
        out = np.zeros((b, c, oh, ow), dtype=np.float32)
        for i in range(oh):
            for j in range(ow):
                out[:, :, i, j] = np.max(x_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw], axis=(-2, -1))
                
        dev = "gpu" if "gpu" in str(x.device) else "cpu"
        return Tensor.from_list(out.flatten().tolist(), shape=list(out.shape)).to(dev)


class AdaptiveAvgPool2d(Module):
    """
    Applies a 2D adaptive average pooling over an input signal composed of several input planes.
    """
    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        target_size = (self.output_size, self.output_size) if isinstance(self.output_size, int) else self.output_size
        
        if target_size == (1, 1):
            # Fast, autograd-safe global pooling using sum_to_shape
            return x.sum_to_shape([b, c, 1, 1]) * (1.0 / (h * w))
            
        # General adaptive average pool fallback
        x_np = np.array(x.to_list(), dtype=np.float32).reshape(x.shape)
        oh, ow = target_size
        out = np.zeros((b, c, oh, ow), dtype=np.float32)
        
        for i in range(oh):
            sh = int(np.floor(i * h / oh))
            eh = int(np.ceil((i + 1) * h / oh))
            for j in range(ow):
                sw = int(np.floor(j * w / ow))
                ew = int(np.ceil((j + 1) * w / ow))
                out[:, :, i, j] = np.mean(x_np[:, :, sh:eh, sw:ew], axis=(-2, -1))
                
        dev = "gpu" if "gpu" in str(x.device) else "cpu"
        return Tensor.from_list(out.flatten().tolist(), shape=list(out.shape)).to(dev)
