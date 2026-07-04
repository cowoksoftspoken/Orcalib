import orca
from orca.nn import Conv2d
from orca.tensor import Tensor
import math

def test_conv2d_advanced():
    print("Testing Conv2d with dilation and groups...")
    
    # 1. Test Dilation
    # Input: 1x1x5x5
    # Weight: 1x1x3x3
    # Dilation: 2
    # Output: 1x1x1x1 (since effective kernel size is 5)
    
    x_data = [float(i) for i in range(25)]
    x = Tensor.from_list(x_data, [1, 1, 5, 5])
    
    w_data = [1.0] * 9
    w = Tensor.from_list(w_data, [1, 1, 3, 3])
    
    conv_dilated = Conv2d(1, 1, 3, dilation=2, bias=False)
    conv_dilated.weight.tensor = w
    
    out_dilated = conv_dilated(x)
    print("Dilated output shape:", out_dilated.shape)
    
    # The output should be the sum of elements at indices:
    # (0,0), (0,2), (0,4)
    # (2,0), (2,2), (2,4)
    # (4,0), (4,2), (4,4)
    # These correspond to values: 0, 2, 4, 10, 12, 14, 20, 22, 24
    # Sum = 108
    print("Dilated output values:", out_dilated.to_list())
    
    assert out_dilated.shape == [1, 1, 1, 1]
    assert abs(out_dilated.to_list()[0] - 108.0) < 1e-4, "Dilation test failed"
    
    # 2. Test Groups
    # Input: 1x4x3x3
    # Weight: 4x2x2x2 (out_channels=4, in_channels_per_group=2, groups=2)
    x_data_g = [float(i) for i in range(36)]
    x_g = Tensor.from_list(x_data_g, [1, 4, 3, 3])
    
    w_data_g = [1.0] * 32 # 4 * 2 * 2 * 2
    w_g = Tensor.from_list(w_data_g, [4, 2, 2, 2])
    
    conv_grouped = Conv2d(4, 4, 2, groups=2, bias=False)
    conv_grouped.weight.tensor = w_g
    
    out_grouped = conv_grouped(x_g)
    print("Grouped output shape:", out_grouped.shape)
    
    assert out_grouped.shape == [1, 4, 2, 2]
    print("Grouped tests passed.")
    print("All advanced Conv2d tests passed!")

if __name__ == "__main__":
    test_conv2d_advanced()
