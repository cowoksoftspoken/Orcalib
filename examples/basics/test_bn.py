import orca
from orca.tensor import Tensor
import orca.nn as nn

def test_batchnorm():
    data = [float(i) for i in range(2 * 3 * 2 * 2)]
    x = Tensor.from_list(data, [2, 3, 2, 2], requires_grad=True)
    
    bn = nn.BatchNorm2d(num_features=3)
    
    # Forward train
    out = bn(x)
    print("Output shape:", out.shape)
    
    # Forward eval
    bn.eval()
    out_eval = bn(x)
    print("Eval output shape:", out_eval.shape)
    
    print("Test passed successfully!")

if __name__ == "__main__":
    test_batchnorm()
