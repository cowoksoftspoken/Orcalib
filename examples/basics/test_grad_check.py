import orca
import orca.nn as nn
from orca.tensor import Tensor
import numpy as np

def compute_numerical_gradient(model, inputs, targets, loss_fn, epsilon=1e-4):
    """
    Computes numerical gradients for a model's parameters using finite differences.
    """
    num_grads = []
    
    for param in model.parameters():
        param_tensor = param.tensor
        shape = param_tensor.shape
        data = np.array(param_tensor.to_list(), dtype=np.float32)
        flat_data = data.flatten()
        grad = np.zeros_like(flat_data)
        
        for i in range(len(flat_data)):
            orig_val = flat_data[i]
            
            # f(x + epsilon)
            flat_data[i] = orig_val + epsilon
            param.update(Tensor.from_list(flat_data.tolist(), shape=list(shape), requires_grad=True))
            out_plus = model(inputs)
            loss_plus = loss_fn(out_plus, targets)
            val_plus = loss_plus.to_list()
            loss_plus_val = val_plus[0] if isinstance(val_plus, list) else val_plus
            
            # f(x - epsilon)
            flat_data[i] = orig_val - epsilon
            param.update(Tensor.from_list(flat_data.tolist(), shape=list(shape), requires_grad=True))
            out_minus = model(inputs)
            loss_minus = loss_fn(out_minus, targets)
            val_minus = loss_minus.to_list()
            loss_minus_val = val_minus[0] if isinstance(val_minus, list) else val_minus
            
            # Restore
            flat_data[i] = orig_val
            
            # df/dx = (f(x+e) - f(x-e)) / (2e)
            grad[i] = (loss_plus_val - loss_minus_val) / (2.0 * epsilon)
            
        param.update(Tensor.from_list(data.flatten().tolist(), shape=list(shape), requires_grad=True))
        num_grads.append(grad.reshape(shape))
        
    return num_grads

def test_linear_relu_grad():
    print("Testing Linear + ReLU Gradients...")
    
    # 1. Setup
    model = nn.Sequential(
        nn.Linear(4, 2),
        nn.ReLU()
    )
    loss_fn = nn.MSELoss()
    
    # 2. Forward & Backward (Analytical)
    x = Tensor.randn([2, 4])
    y = Tensor.randn([2, 2])
    
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    
    analytical_grads = []
    for p in model.parameters():
        agrad = p.tensor.grad()
        if agrad is not None:
            analytical_grads.append(np.array(agrad.to_list()).reshape(p.tensor.shape))
        else:
            analytical_grads.append(np.zeros(p.tensor.shape))
            
    # 3. Numerical Gradients
    numerical_grads = compute_numerical_gradient(model, x, y, loss_fn)
    
    # 4. Compare
    for i, (a_grad, n_grad) in enumerate(zip(analytical_grads, numerical_grads)):
        diff = np.abs(a_grad - n_grad)
        max_diff = np.max(diff)
        print(f"  Param {i} max diff: {max_diff:.6f}")
        assert max_diff < 5e-3, f"Gradient mismatch on param {i}! Max diff: {max_diff}"
        
    print("Gradient Check Passed!\n")

def test_cross_entropy_grad():
    print("Testing Linear + CrossEntropy Gradients...")
    
    model = nn.Linear(3, 3)
    loss_fn = nn.CrossEntropyLoss()
    
    x_data = np.random.randn(2, 3).astype(np.float32)
    x = Tensor.from_list(x_data.flatten().tolist(), shape=[2, 3], requires_grad=False)
    # One hot targets
    y_arr = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    y = Tensor.from_list(y_arr.flatten().tolist(), shape=[2, 3], requires_grad=False)
    
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    
    analytical_grads = []
    for p in model.parameters():
        agrad = p.tensor.grad()
        if agrad is not None:
            analytical_grads.append(np.array(agrad.to_list()).reshape(p.tensor.shape))
        else:
            analytical_grads.append(np.zeros(p.tensor.shape))
            
    numerical_grads = compute_numerical_gradient(model, x, y, loss_fn)
    
    for i, (a_grad, n_grad) in enumerate(zip(analytical_grads, numerical_grads)):
        diff = np.abs(a_grad - n_grad)
        max_diff = np.max(diff)
        print(f"  Param {i} max diff: {max_diff:.6f}")
        assert max_diff < 5e-3, f"Gradient mismatch on param {i}! Max diff: {max_diff}"
        
    print("Gradient Check Passed!\n")

def test_layernorm_grad():
    print("Testing LayerNorm Gradients...")
    model = nn.LayerNorm(4)
    loss_fn = nn.MSELoss()
    
    x = Tensor.randn([2, 4], requires_grad=False)
    y = Tensor.randn([2, 4], requires_grad=False)
    
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    
    analytical_grads = []
    for p in model.parameters():
        agrad = p.tensor.grad()
        if agrad is not None:
            analytical_grads.append(np.array(agrad.to_list()).reshape(p.tensor.shape))
        else:
            analytical_grads.append(np.zeros(p.tensor.shape))
            
    numerical_grads = compute_numerical_gradient(model, x, y, loss_fn)
    
    for i, (a_grad, n_grad) in enumerate(zip(analytical_grads, numerical_grads)):
        diff = np.abs(a_grad - n_grad)
        max_diff = np.max(diff)
        print(f"  Param {i} max diff: {max_diff:.6f}")
        assert max_diff < 5e-3, f"Gradient mismatch on param {i}! Max diff: {max_diff}"
        
    print("Gradient Check Passed!\n")

def test_batchnorm_grad():
    print("Testing BatchNorm2d Gradients...")
    model = nn.BatchNorm2d(3)
    loss_fn = nn.MSELoss()
    
    x = Tensor.randn([2, 3, 4, 4], requires_grad=False)
    y = Tensor.randn([2, 3, 4, 4], requires_grad=False)
    
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    
    analytical_grads = []
    for p in model.parameters():
        agrad = p.tensor.grad()
        if agrad is not None:
            analytical_grads.append(np.array(agrad.to_list()).reshape(p.tensor.shape))
        else:
            analytical_grads.append(np.zeros(p.tensor.shape))
            
    numerical_grads = compute_numerical_gradient(model, x, y, loss_fn, epsilon=1e-3)
    
    for i, (a_grad, n_grad) in enumerate(zip(analytical_grads, numerical_grads)):
        diff = np.abs(a_grad - n_grad)
        max_diff = np.max(diff)
        print(f"  Param {i} max diff: {max_diff:.6f}")
        # BatchNorm involves many operations, float32 precision can cause drift
        assert max_diff < 2e-2, f"Gradient mismatch on param {i}! Max diff: {max_diff}"
        
    print("Gradient Check Passed!\n")

def test_conv2d_grad():
    print("Testing Conv2d Gradients...")
    model = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, padding=1)
    loss_fn = nn.MSELoss()
    
    x = Tensor.randn([2, 2, 4, 4], requires_grad=False)
    y = Tensor.randn([2, 3, 4, 4], requires_grad=False)
    
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    
    analytical_grads = []
    for p in model.parameters():
        agrad = p.tensor.grad()
        if agrad is not None:
            analytical_grads.append(np.array(agrad.to_list()).reshape(p.tensor.shape))
        else:
            analytical_grads.append(np.zeros(p.tensor.shape))
            
    numerical_grads = compute_numerical_gradient(model, x, y, loss_fn, epsilon=1e-3)
    
    for i, (a_grad, n_grad) in enumerate(zip(analytical_grads, numerical_grads)):
        diff = np.abs(a_grad - n_grad)
        max_diff = np.max(diff)
        print(f"  Param {i} max diff: {max_diff:.6f}")
        assert max_diff < 5e-2, f"Gradient mismatch on param {i}! Max diff: {max_diff}"
        
    print("Gradient Check Passed!\n")

if __name__ == "__main__":
    test_linear_relu_grad()
    test_cross_entropy_grad()
    test_layernorm_grad()
    test_batchnorm_grad()
    test_conv2d_grad()
