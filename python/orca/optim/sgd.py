from .optimizer import Optimizer
from orca.tensor import Tensor

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    """
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self):
        """
        Clears the global computational graph and all gradients.
        """
        if len(self.parameters) > 0:
            self.parameters[0].tensor.zero_grad()

    def step(self):
        """
        Performs a single optimization step.
        """
        for param in self.parameters:
            grad = param.tensor.grad()
            if grad is not None:
                # Calculate the update: grad * lr
                update = grad * self.lr
                
                # Perform gradient descent step: param = param - update
                new_tensor_tracked = param.tensor - update
                
                # We want to treat the updated parameter as a new leaf node,
                # breaking the computational graph history.
                # Copy the data out to a list and create a new tensor.
                new_data = new_tensor_tracked.to_list()
                
                # We create a new tensor with the updated data and tell the engine to track it
                new_leaf_tensor = Tensor.from_list(
                    new_data, 
                    shape=param.tensor.shape, 
                    requires_grad=True
                )
                
                # Update the parameter's internal reference
                param.update(new_leaf_tensor)
