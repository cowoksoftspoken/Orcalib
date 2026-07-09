from typing import List
from .orca_python import Tensor as _Tensor, Device, DType

class Tensor(_Tensor):
    # Wrap the PyO3 tensor to add pure Python utility methods
    def transpose(self, dim0: int, dim1: int) -> 'Tensor':
        # Actually PyO3 Tensor methods return the PyO3 Tensor, so we need to wrap them back
        # Wait, if we subclass `_Tensor`, the methods returning `Self` in Rust will return `_Tensor` instances in Python!
        # So `self.reshape()` returns `_Tensor`. We shouldn't subclass it like this if Rust returns the base class.
        pass

# It's better to add the methods dynamically to the PyO3 class if we want to extend it, 
# or just expose standalone functions.

def einsum(equation: str, *operands: _Tensor) -> _Tensor:
    # Delegate to native Rust einsum parser via _Tensor static method
    return _Tensor.einsum(equation, list(operands))

__all__ = ["Tensor", "Device", "DType", "einsum"]
Tensor = _Tensor
