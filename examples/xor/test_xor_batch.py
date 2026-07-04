import orca
import orca.nn as nn
import orca.optim as optim

# XOR Neural Network
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

optimizer = optim.SGD(model.parameters(), lr=0.5)
loss_fn = nn.MSELoss()

# XOR Dataset (Full Batch)
# Shape: [4, 2]
X_batch = orca.Tensor.from_list([
    0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0
], shape=[4, 2])

# Shape: [4, 1]
Y_batch = orca.Tensor.from_list([
    0.0,
    1.0,
    1.0,
    0.0
], shape=[4, 1])

print("Training XOR model with Full Batch (size 4)...")
for epoch in range(1000):
    optimizer.zero_grad()
    
    # Forward pass on the entire batch at once!
    y_pred = model(X_batch)
    
    # Loss automatically computes mean over the batch
    loss = loss_fn(y_pred, Y_batch)
    
    # Backward pass aggregates gradients over the batch
    loss.backward()
    
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.to_list()[0]:.4f}")

print("\nFinal Predictions:")
y_pred = model(X_batch)
out_list = y_pred.to_list()
print(f"Input [0,0] => Pred: {out_list[0]:.4f}")
print(f"Input [0,1] => Pred: {out_list[1]:.4f}")
print(f"Input [1,0] => Pred: {out_list[2]:.4f}")
print(f"Input [1,1] => Pred: {out_list[3]:.4f}")
