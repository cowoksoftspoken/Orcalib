import orca
import orca.nn as nn
import orca.optim as optim
import random

# XOR Neural Network
# Using a larger hidden layer to avoid local minima
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

# XOR Dataset
X = [
    orca.Tensor.from_list([0.0, 0.0], shape=[1, 2]),
    orca.Tensor.from_list([0.0, 1.0], shape=[1, 2]),
    orca.Tensor.from_list([1.0, 0.0], shape=[1, 2]),
    orca.Tensor.from_list([1.0, 1.0], shape=[1, 2]),
]
Y = [
    orca.Tensor.from_list([0.0], shape=[1, 1]),
    orca.Tensor.from_list([1.0], shape=[1, 1]),
    orca.Tensor.from_list([1.0], shape=[1, 1]),
    orca.Tensor.from_list([0.0], shape=[1, 1]),
]

print("Training XOR model...")
model.to('cpu')
X = [x.to('cpu') for x in X]
Y = [y.to('cpu') for y in Y]
for epoch in range(2000):
    total_loss = 0.0
    
    # Shuffle for SGD
    indices = [0, 1, 2, 3]
    random.shuffle(indices)
    
    for i in indices:
        optimizer.zero_grad()
        y_pred = model(X[i])
        loss = loss_fn(y_pred, Y[i])
        loss.backward()
        optimizer.step()
        total_loss += loss.to_list()[0]
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}: Loss = {total_loss/4:.4f}")

print("\nFinal Predictions:")
for i in range(4):
    y_pred = model(X[i])
    print(f"Input: {X[i].to_list()} => Pred: {y_pred.to_list()[0]:.4f} (True: {Y[i].to_list()[0]})")
