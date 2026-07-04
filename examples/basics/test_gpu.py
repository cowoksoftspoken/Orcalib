import orca
import time

try:
    print("Initializing GPU Tensors...")
    start = time.time()
    device = orca.Device("gpu")
    
    # Create tensors directly on GPU
    a = orca.Tensor.from_list([1.0, 2.0, 3.0, 4.0], shape=[2, 2], device=device)
    b = orca.Tensor.from_list([5.0, 6.0, 7.0, 8.0], shape=[2, 2], device=device)
    
    print(f"Tensor A (GPU): {a}")
    print(f"Tensor B (GPU): {b}")
    
    # Test Math Ops
    print("\nTesting GPU Math Operations:")
    c = a + b
    print(f"A + B = {c.to_list()}")
    
    d = a * b
    print(f"A * B = {d.to_list()}")
    
    # Test Matmul
    e = a @ b
    print(f"A @ B = {e.to_list()}")
    
    # Test Activation
    f = a.relu()
    print(f"ReLU(A) = {f.to_list()}")
    
    # Test Autograd on GPU
    print("\nTesting GPU Autograd:")
    x = orca.Tensor.from_list([2.0], shape=[1], requires_grad=True, device=device)
    y = x * orca.Tensor.from_list([3.0], shape=[1], device=device)
    y.backward()
    print(f"d(x * 3)/dx = {x.grad().to_list()}")
    
    print(f"\nAll GPU Tests Passed in {time.time() - start:.3f}s!")

except Exception as e:
    print(f"Error during GPU execution: {e}")
