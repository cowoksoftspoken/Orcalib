from typing import List
from .orca_python import Tensor, Device, DType

def einsum(equation: str, *operands: Tensor) -> Tensor:
    """
    Evaluates the Einstein summation convention on the operands.
    
    Args:
        equation (str): The equation string in Einstein summation convention.
        *operands (Tensor): The tensors to compute the einsum for.
        
    Returns:
        Tensor: The evaluated tensor.
    """
    return Tensor.einsum(equation, list(operands))

__all__ = ["Tensor", "Device", "DType", "einsum"]
