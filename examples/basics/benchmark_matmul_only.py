import orca
from orca.tensor import Tensor, Device
import time

def benchmark_matmul(size, num_steps=100):
    device = Device("gpu")
    
    a = Tensor.ones([size, size], device=device)
    b = Tensor.ones([size, size], device=device)
    
    # Warmup
    for _ in range(5):
        c = a @ b
        
    start = time.time()
    for _ in range(num_steps):
        c = a @ b
        
    end = time.time()
    total = end - start
    print(f"Matrix Size: {size}x{size}")
    print(f"Time per matmul: {(total/num_steps)*1000:.2f} ms")

if __name__ == "__main__":
    benchmark_matmul(128)
    benchmark_matmul(512)
    benchmark_matmul(1024)
