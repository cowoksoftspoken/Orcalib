class Parameter:
    """
    A kind of Tensor that is to be considered a module parameter.
    """
    def __init__(self, tensor):
        self.tensor = tensor
        # We rely on the tensor already being created with requires_grad=True

    def update(self, new_tensor):
        """
        In-place update of the parameter's tensor data (used by Optimizers).
        """
        self.tensor = new_tensor

    def to(self, device):
        """
        Move parameter to device (e.g. 'cpu', 'gpu').
        """
        self.tensor = self.tensor.to(device)
        return self
