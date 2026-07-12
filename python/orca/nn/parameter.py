from typing import Any, Dict, Iterator, Optional, Union
from orca.tensor import Tensor
import orca

class Parameter:
    """
    A kind of Tensor that is to be considered a module parameter.
    
    Parameters are `Tensor` subclasses that have a very special property when used with
    `Module`s - when they're assigned as Module attributes they are automatically added
    to the list of its parameters, and will appear e.g. in `parameters()` iterator.
    """
    def __init__(self, tensor: Tensor):
        """
        Args:
            tensor (Tensor): The tensor to be wrapped as a parameter.
        """
        self.tensor = tensor

    def update(self, new_tensor: Tensor) -> None:
        """
        In-place update of the parameter's tensor data.
        Typically used by Optimizers during the `step()` phase.
        
        Args:
            new_tensor (Tensor): The new tensor to replace the current parameter data.
        """
        self.tensor = new_tensor

    def to(self, device: str) -> 'Parameter':
        """
        Moves the parameter to a specified device (e.g., 'cpu' or 'gpu').
        
        Args:
            device (str): The target device.
            
        Returns:
            Parameter: self
        """
        self.tensor = self.tensor.to(device)
        return self
