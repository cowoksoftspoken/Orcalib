class Optimizer:
    """
    Base class for all optimizers.
    """
    def __init__(self, parameters):
        self.parameters = list(parameters)

    def zero_grad(self):
        for param in self.parameters:
            if param.tensor.requires_grad:
                param.tensor.zero_grad()

    def step(self):
        raise NotImplementedError
