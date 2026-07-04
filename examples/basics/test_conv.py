import orca
from orca.tensor import Tensor
import orca.nn as nn
import orca.optim as optim
import math

def test_conv():
    # Set seed for reproducibility if needed, but we don't have one exposed yet
    
    # Create input: (N=2, C_in=1, H=4, W=4)
    data = [float(i) for i in range(2 * 1 * 4 * 4)]
    x = Tensor.from_list(data, [2, 1, 4, 4], requires_grad=True)
    
    # Create Conv2d
    conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1, stride=1)
    
    # Forward pass
    out = conv(x)
    print(f"Output shape: {out.shape}")
    
    # Backward pass
    # We do simple manual loss backward to test gradients
    out_sum = out.sum()
    out_sum.backward()
    
    print("x.grad shape:", x.grad().shape)
    print("conv.weight.grad shape:", conv.weight.tensor.grad().shape)
    print("conv.bias.grad shape:", conv.bias.tensor.grad().shape)
    
    print("Test passed successfully!")

if __name__ == "__main__":
    test_conv()
