import orca

def clip_grad_value_(parameters, clip_value: float):
    """
    Clips gradient of an iterable of parameters at specified value.
    Gradients are modified in-place.
    """
    clip_value = float(clip_value)
    for p in parameters:
        if p.tensor.requires_grad:
            grad = p.tensor.grad()
            if grad is not None:
                # clamp grad between -clip_value and clip_value
                # Assuming orca backend has clamp, or we can use max/min manually
                
                # Manual clamp using conditional masks if clamp is not available:
                # But we can just use max/min if they are available
                # grad = grad.maximum(-clip_value).minimum(clip_value)
                
                # Since we haven't checked if minimum/maximum exists on Tensor, 
                # a more robust way for now is to use ReLU:
                # clamp(x, min, max) = min + relu(x - min) - relu(x - max)
                min_tensor = orca.Tensor.scalar(-clip_value, device=grad.device)
                max_tensor = orca.Tensor.scalar(clip_value, device=grad.device)
                
                grad = min_tensor + (grad - min_tensor).relu() - (grad - max_tensor).relu()
                p.tensor._grad = grad

def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0):
    """
    Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    """
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    
    total_norm = 0.0
    valid_params = []
    
    for p in parameters:
        if p.tensor.requires_grad:
            grad = p.tensor.grad()
            if grad is not None:
                valid_params.append((p, grad))
                
                # Flatten grad to 1D equivalent by summing powers
                # grad_norm = (grad ** 2).sum()
                grad_sq = grad * grad
                sum_shape = [1] * len(grad_sq.shape)
                summed = grad_sq.sum_to_shape(sum_shape)
                
                # For a scalar, to_list()[0] works if it's 1D, or we can flatten
                flat_val = summed.to_list()
                while isinstance(flat_val, list):
                    flat_val = flat_val[0]
                    
                total_norm += flat_val
                
    total_norm = total_norm ** (1. / norm_type)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p, grad in valid_params:
            # We must recreate a tensor or scale in place
            # scale = orca.Tensor.scalar(clip_coef, device=grad.device)
            # p.tensor._grad = grad * scale
            
            # Since scale is a scalar float, we can create a scalar tensor:
            scale_tensor = orca.Tensor.scalar(clip_coef, device=grad.device)
            p.tensor._grad = grad * scale_tensor
            
    return total_norm
