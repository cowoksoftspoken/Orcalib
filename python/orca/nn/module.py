from .parameter import Parameter
import orca
import numpy as np
from safetensors.numpy import save_file, load_file

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
        state = self.state_dict()
        # Convert flat lists to properly shaped numpy arrays for safetensors
        np_state = {}
        for name, param in self._parameters.items():
            shape = param.tensor.shape
            np_state[name] = np.array(state[name], dtype=np.float32).reshape(shape)
        
        for name, module in self._modules.items():
            module_state = module.state_dict(prefix=name + '.')
            for k, v in module_state.items():
                # We need to find the shape of the parameter inside the submodule
                # This requires a bit of digging, or we can just save it flat for now.
                # Actually, safetensors requires contiguous arrays and shape.
                # Let's get the shape from the actual parameter objects!
                pass
        
        # A better approach: gather all parameters directly with their names
        np_state = {}
        def _gather_params(mod, prefix=''):
            for n, p in mod._parameters.items():
                np_state[prefix + n] = np.array(p.tensor.to_list(), dtype=np.float32).reshape(p.tensor.shape)
            for n, b in mod._buffers.items():
                if isinstance(b, orca.Tensor):
                    np_state[prefix + n] = np.array(b.to_list(), dtype=np.float32).reshape(b.shape)
                else:
                    np_state[prefix + n] = np.array([b], dtype=np.float32)
            for n, m in mod._modules.items():
                _gather_params(m, prefix + n + '.')
        
        _gather_params(self)
        save_file(np_state, filepath)

    def load_weights(self, filepath):
        np_state = load_file(filepath)
        state = {k: v.flatten().tolist() for k, v in np_state.items()}
        self.load_state_dict(state)
