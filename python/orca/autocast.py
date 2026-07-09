class autocast:
    """
    Context manager that enables mixed precision for operations.
    Since Orca currently supports f32 everywhere, this is mostly a placeholder
    to maintain API compatibility with PyTorch until f16/bf16 kernels are fully integrated.
    """
    def __init__(self, device_type='cuda', dtype=None, enabled=True):
        self.device_type = device_type
        self.dtype = dtype
        self.enabled = enabled
        
    def __enter__(self):
        if self.enabled:
            pass
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            pass

class GradScaler:
    """
    Gradient scaling for mixed precision training.
    """
    def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.enabled = enabled
        self._step_count = 0

    def scale_loss(self, loss):
        if not self.enabled:
            return loss
        # Since loss is an orca.Tensor, we can multiply it
        import orca
        scale_tensor = orca.Tensor.scalar(self.scale, device=loss.device)
        return loss * scale_tensor
        
    def scale_tensor(self, tensor):
        # A wrapper for scale_loss in case people call scale(loss)
        return self.scale_loss(tensor)

    def step(self, optimizer):
        if not self.enabled:
            optimizer.step()
            return
            
        # Check for inf/nan in gradients
        found_inf = False
        for param in optimizer.params:
            if param.grad is not None:
                if param.grad.has_nan_or_inf():
                    found_inf = True
                    break
                    
        if not found_inf:
            optimizer.step()
        self._found_inf = found_inf

    def update(self):
        if not self.enabled:
            return
            
        if getattr(self, '_found_inf', False):
            self.scale *= self.backoff_factor
            self._step_count = 0
        else:
            self._step_count += 1
            if self._step_count >= self.growth_interval:
                self.scale *= self.growth_factor
                self._step_count = 0
