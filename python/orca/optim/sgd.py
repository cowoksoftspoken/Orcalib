"""Stochastic Gradient Descent optimizer with momentum and weight decay."""
from typing import Iterable, Optional
import orca
from .optimizer import Optimizer
from orca.nn.parameter import Parameter


class SGD(Optimizer):
    """Implements stochastic gradient descent with optional momentum and weight decay.

    Nesterov momentum is **not** supported in this version.

    The update rule (with momentum and weight decay) is::

        v_t = momentum * v_{t-1} + grad + weight_decay * param
        param = param - lr * v_t

    Args:
        parameters: Iterable of parameters to optimize.
        lr: Learning rate. Default: ``0.01``.
        momentum: Momentum factor. Default: ``0.0`` (vanilla SGD).
        weight_decay: L2 penalty coefficient. Default: ``0.0`` (no penalty).
        dampening: Dampening for momentum. Default: ``0.0``.
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
    ):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening

        # Velocity buffers (lazy-initialized on first step)
        self._velocity = [None] * len(self.parameters)

    def step(self) -> None:
        """Performs a single optimization step."""
        for i, param in enumerate(self.parameters):
            grad = param.tensor.grad()
            if grad is None:
                continue

            device = param.tensor.device

            # L2 weight decay: d_p = grad + weight_decay * param
            if self.weight_decay != 0.0:
                grad = grad + param.tensor * self.weight_decay

            # Momentum
            if self.momentum != 0.0:
                if self._velocity[i] is None:
                    # First call — clone the gradient as initial velocity
                    self._velocity[i] = grad.detach()
                else:
                    # v_t = momentum * v_{t-1} + (1 - dampening) * grad
                    v_prev = self._velocity[i]
                    if self.dampening != 0.0:
                        self._velocity[i] = v_prev * self.momentum + grad * (1.0 - self.dampening)
                    else:
                        self._velocity[i] = v_prev * self.momentum + grad

                grad = self._velocity[i]

            # param = param - lr * grad
            update = grad * self.lr
            new_tensor = param.tensor - update

            # Detach from graph and re-enable gradient tracking (zero-copy)
            new_leaf = new_tensor.detach()
            new_leaf.require_grad()
            param.update(new_leaf)
