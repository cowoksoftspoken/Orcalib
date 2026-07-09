from .parameter import Parameter
import orca
import orca
from orca.tensor import Tensor

class Module:
    """
    Base class for all neural network modules.
    """
    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self._buffers = {}
        self.training = True

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            if not hasattr(self, '_parameters'):
                self._parameters = {}
            self._parameters[name] = value
        elif isinstance(value, Module):
            if not hasattr(self, '_modules'):
                self._modules = {}
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        """
        Returns an iterator over module parameters.
        """
        for name, param in self._parameters.items():
            yield param
        for name, module in self._modules.items():
            for param in module.parameters():
                yield param

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def to(self, device):
        """Moves all parameters and buffers to the specified device."""
        for param in self._parameters.values():
            param.to(device)
        for buf_name, buf in self._buffers.items():
            if isinstance(buf, orca.Tensor):
                self._buffers[buf_name] = buf.to(device)
        for module in self._modules.values():
            module.to(device)
        return self

    def train(self, mode: bool = True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def register_buffer(self, name, tensor):
        self._buffers[name] = tensor

    def state_dict(self, prefix=''):
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

    def load_state_dict(self, state):
        for name, param in self._parameters.items():
            if name in state:
                # Reconstruct tensor on the same device but with loaded values
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

    def save_weights(self, filepath):
        # Gather all parameters natively as PyTensor
        state = {}
        def _gather_params(mod, prefix=''):
            for n, p in mod._parameters.items():
                state[prefix + n] = p.tensor
            for n, b in mod._buffers.items():
                if isinstance(b, orca.Tensor):
                    state[prefix + n] = b
            for n, m in mod._modules.items():
                _gather_params(m, prefix + n + '.')
        
        _gather_params(self)
        orca.save_tensors(filepath, state)

    def load_weights(self, filepath):
        state = orca.load_tensors(filepath)
        
        # Apply the loaded tensors directly to the parameters
        def _apply_params(mod, prefix=''):
            for n, p in mod._parameters.items():
                key = prefix + n
                if key in state:
                    p.tensor = state[key]
            for n, b in mod._buffers.items():
                key = prefix + n
                if key in state:
                    if isinstance(b, orca.Tensor):
                        mod._buffers[n] = state[key]
            for n, m in mod._modules.items():
                _apply_params(m, prefix + n + '.')
                
        _apply_params(self)
