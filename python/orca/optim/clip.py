"""Gradient clipping utilities.

These functions modify gradients stored on the autograd tape in-place,
exactly like ``torch.nn.utils.clip_grad_norm_`` and ``clip_grad_value_``.
"""
from typing import Iterable
import orca
from orca.nn.parameter import Parameter


def clip_grad_value_(parameters: Iterable[Parameter], clip_value: float) -> None:
    """Clips gradient of an iterable of parameters at specified value.

    Gradients are modified in-place on the tape. After calling this,
    every element of every gradient will be in ``[-clip_value, clip_value]``.

    Args:
        parameters: Iterable of ``Parameter`` whose ``.tensor`` may carry a gradient.
        clip_value: Maximum absolute value for any gradient element.
    """
    clip_value = float(clip_value)
    for p in parameters:
        if p.tensor.requires_grad:
            grad = p.tensor.grad()
            if grad is None:
                continue

            # clamp(x, -c, c) = -c + relu(x - (-c)) - relu(x - c)
            min_t = orca.Tensor.scalar(-clip_value, device=grad.device)
            max_t = orca.Tensor.scalar(clip_value, device=grad.device)

            clamped = min_t + (grad - min_t).relu() - (grad - max_t).relu()

            # Write the clamped gradient back into the autograd tape
            p.tensor.set_grad(clamped)


def clip_grad_norm_(
    parameters: Iterable[Parameter],
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters: Iterable of ``Parameter`` whose ``.tensor`` may carry a gradient.
        max_norm: Maximum allowed norm value.
        norm_type: Type of the p-norm (default: L2).

    Returns:
        Total (unclipped) norm of the gradients as a Python float.
    """
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    total_norm_sq = 0.0
    valid_params = []

    for p in parameters:
        if p.tensor.requires_grad:
            grad = p.tensor.grad()
            if grad is None:
                continue

            valid_params.append((p, grad))

            # ||grad||_2^2 = sum(grad^2)
            grad_sq = grad * grad
            sum_shape = [1] * len(grad_sq.shape)
            summed = grad_sq.sum_to_shape(sum_shape)

            flat = summed.to_list()
            while isinstance(flat, list):
                flat = flat[0]

            total_norm_sq += float(flat)

    total_norm = total_norm_sq ** (1.0 / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p, grad in valid_params:
            scale = orca.Tensor.scalar(clip_coef, device=grad.device)
            scaled_grad = grad * scale
            # Write the scaled gradient back into the autograd tape
            p.tensor.set_grad(scaled_grad)

    return total_norm
