import orca
import orca.nn as nn
import orca.optim as optim

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 1)
        
    def forward(self, x):
        return self.fc1(x)

model = MyModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

x = orca.Tensor.from_list([1.0, 2.0], shape=[1, 2])
y_true = orca.Tensor.from_list([5.0], shape=[1, 1])

print("Initial weights:", model.fc1.weight.tensor.to_list())
print("Initial bias:", model.fc1.bias.tensor.to_list())

for epoch in range(10):
    optimizer.zero_grad()
    
    y_pred = model(x)
    
    # MSE Loss: (y_pred - y_true) * (y_pred - y_true)
    diff = y_pred - y_true
    loss = diff @ diff
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}: Loss = {loss.to_list()[0]}")

print("Final weights:", model.fc1.weight.tensor.to_list())
print("Final bias:", model.fc1.bias.tensor.to_list())
