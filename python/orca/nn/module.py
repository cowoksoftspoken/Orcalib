from typing import Any, Dict, Iterator, Optional, Union
from .parameter import Parameter
import orca
from orca.tensor import Tensor

class Module:
    """
    Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in a tree structure.
    You can assign the submodules as regular attributes.
    """
    def __init__(self) -> None:
        self._modules: Dict[str, 'Module'] = {}
        self._parameters: Dict[str, Parameter] = {}
        self._buffers: Dict[str, Any] = {}
        self.training: bool = True

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            if not hasattr(self, '_parameters'):
                self._parameters = {}
            self._parameters[name] = value
        elif isinstance(value, Module):
            if not hasattr(self, '_modules'):
                self._modules = {}
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self) -> Iterator[Parameter]:
        """
        Returns an iterator over module parameters.
        This is typically passed to an optimizer.
        
        Yields:
            Parameter: module parameter
        """
        for name, param in self._parameters.items():
            yield param
        for name, module in self._modules.items():
            for param in module.parameters():
                yield param

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def to(self, device: str) -> 'Module':
        """
        Moves all parameters and buffers to the specified device.
        
        Args:
            device (str): The target device ('cpu' or 'gpu').
            
        Returns:
            Module: self
        """
        for param in self._parameters.values():
            param.to(device)
        for buf_name, buf in self._buffers.items():
            if isinstance(buf, orca.Tensor):
                self._buffers[buf_name] = buf.to(device)
        for module in self._modules.values():
            module.to(device)
        return self

    def train(self, mode: bool = True) -> 'Module':
        """
        Sets the module in training mode.
        This has any effect only on certain modules (e.g. Dropout, BatchNorm).
        
        Args:
            mode (bool): whether to set training mode (True) or evaluation mode (False). Defaults to True.
            
        Returns:
            Module: self
        """
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> 'Module':
        """
        Sets the module in evaluation mode.
        This is equivalent to `self.train(False)`.
        
        Returns:
            Module: self
        """
        return self.train(False)

    def register_buffer(self, name: str, tensor: Any) -> None:
        """
        Adds a buffer to the module.
        This is typically used to register a state that should not to be considered a model parameter.
        
        Args:
            name (str): name of the buffer.
            tensor (Any): buffer object to be registered.
        """
        self._buffers[name] = tensor

    def state_dict(self, prefix: str = '') -> Dict[str, Any]:
        """
        Returns a dictionary containing a whole state of the module.
        
        Args:
            prefix (str): optional prefix for the keys.
            
        Returns:
            Dict[str, Any]: a dictionary containing the state.
        """
        state = {}
        for name, param in self._parameters.items():
            state[prefix + name] = param.tensor.to_list()
        for name, buf in self._buffers.items():
            if isinstance(buf, orca.Tensor):
                state[prefix + name] = buf.to_list()
            else:
                state[prefix + name] = buf
        for name, module in self._modules.items():
            state.update(module.state_dict(prefix + name + '.'))
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Copies parameters and buffers from `state` into this module and its descendants.
        
        Args:
            state (Dict[str, Any]): a dict containing parameters and persistent buffers.
        """
        for name, param in self._parameters.items():
            if name in state:
                device_str = "gpu" if "gpu" in str(param.tensor.device) else "cpu"
                device = orca.Device(device_str)
                param.tensor = orca.Tensor.from_list(
                    state[name], 
                    shape=param.tensor.shape, 
                    requires_grad=param.tensor.requires_grad, 
                    device=device
                )
        for name, buf in self._buffers.items():
            if name in state:
                if isinstance(buf, orca.Tensor):
                    device_str = "gpu" if "gpu" in str(buf.device) else "cpu"
                    device = orca.Device(device_str)
                    self._buffers[name] = orca.Tensor.from_list(
                        state[name],
                        shape=buf.shape,
                        requires_grad=False,
                        device=device
                    )
                else:
                    self._buffers[name] = state[name]

        for name, module in self._modules.items():
            module_state = {k.replace(f"{name}.", "", 1): v for k, v in state.items() if k.startswith(f"{name}.")}
            module.load_state_dict(module_state)

    def save_weights(self, filepath: str) -> None:
        """
        Saves the module parameters to a file in Safetensors format.
        
        Args:
            filepath (str): the path to save the weights to.
        """
        state = {}
        def _gather_params(mod: 'Module', prefix: str = '') -> None:
            for n, p in mod._parameters.items():
                state[prefix + n] = p.tensor
            for n, b in mod._buffers.items():
                if isinstance(b, orca.Tensor):
                    state[prefix + n] = b
            for n, m in mod._modules.items():
                _gather_params(m, prefix + n + '.')
        
        _gather_params(self)
        orca.save_tensors(filepath, state)

    def load_weights(self, filepath: str) -> None:
        """
        Loads the module parameters from a file in Safetensors format.
        
        Args:
            filepath (str): the path to load the weights from.
        """
        state = orca.load_tensors(filepath)
        
        def _apply_params(mod: 'Module', prefix: str = '') -> None:
            for n, p in mod._parameters.items():
                key = prefix + n
                if key in state:
                    device_str = str(p.tensor.device).lower()
                    device = "gpu" if "gpu" in device_str else "cpu"
                    orig_requires_grad = p.tensor.requires_grad
                    p.tensor = state[key].to(device)
                    if orig_requires_grad:
                        p.tensor.require_grad()
            for n, b in mod._buffers.items():
                key = prefix + n
                if key in state:
                    if isinstance(b, orca.Tensor):
                        device_str = str(b.device).lower()
                        device = "gpu" if "gpu" in device_str else "cpu"
                        mod._buffers[n] = state[key].to(device)
            for n, m in mod._modules.items():
                _apply_params(m, prefix + n + '.')
                
        _apply_params(self)
