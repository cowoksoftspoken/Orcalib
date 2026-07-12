"""Base class for all optimizers."""
from typing import Iterable
from orca.nn.parameter import Parameter


class Optimizer:
    """Base class for all optimizers.

    Args:
        parameters: An iterable of ``Parameter`` objects to optimize.
    """

    def __init__(self, parameters: Iterable[Parameter]):
        self.parameters = list(parameters)

    def zero_grad(self) -> None:
        """Clears the computational graph and all accumulated gradients.

        This resets the autograd tape so that the next forward pass
        builds a fresh computation graph. Only one call to the underlying
        tape clear is needed since all parameters share the same tape.
        """
        if self.parameters:
            self.parameters[0].tensor.zero_grad()

    def step(self) -> None:
        """Performs a single optimization step (parameter update).

        Must be implemented by subclasses.

        Raises:
            NotImplementedError: Always, unless overridden.
        """
        raise NotImplementedError
