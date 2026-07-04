# Re-export core types from the compiled Rust extension.
import math
import random
from typing import List, Optional
from .orca_python import DType, Device

def randn(shape: List[int], requires_grad: bool = False):
    """
    Creates a tensor with random numbers from a standard normal distribution.
    """
    from .tensor import Tensor
    
    num_elements = 1
    for dim in shape:
        num_elements *= dim
        
    data = [random.gauss(0.0, 1.0) for _ in range(num_elements)]
    return Tensor.from_list(data, shape=shape, requires_grad=requires_grad)

__all__ = ["DType", "Device", "randn"]
