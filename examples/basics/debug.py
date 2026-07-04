import orca
import orca.nn as nn
from orca.tensor import Tensor
import numpy as np

# Let's run a single forward/backward pass and print gradients
model = nn.Linear(2, 2)
loss_fn = nn.CrossEntropyLoss()

X = Tensor.from_list([0.5, -0.5], shape=[1, 2], requires_grad=True)
y = Tensor.from_list([1.0, 0.0], shape=[1, 2], requires_grad=False)

print("Forward pass...")
pred = model(X)
print("Pred:", pred.to_list())
loss = loss_fn(pred, y)
print("Loss:", loss.to_list())

print("Backward pass...")
loss.backward()

print("Weight Grad:", model.weight.tensor.grad().to_list())
print("Bias Grad:", model.bias.tensor.grad().to_list())

