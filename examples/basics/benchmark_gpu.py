import orca
from orca.tensor import Tensor, Device
import time

def run_benchmark(shape_in, shape_out, batch_size, num_steps=50):
    device = Device("gpu")
    
    # Initialize tensors on GPU using our new pythonic API!
    x = Tensor.randn([batch_size, shape_in], device=device, requires_grad=False)
    w = Tensor.randn([shape_in, shape_out], device=device, requires_grad=True)
    b = Tensor.randn([shape_out], device=device, requires_grad=True)
    target = Tensor.randn([batch_size, shape_out], device=device, requires_grad=False)
    
    # Warmup
    for _ in range(3):
        out = (x @ w + b).relu()
        diff = out - target
        loss = (diff * diff).sum()
        loss.backward()
        
    start_time = time.time()
    for _ in range(num_steps):
        out = (x @ w + b).relu()
        diff = out - target
        loss = (diff * diff).sum()
        loss.backward()
        
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_step = (total_time / num_steps) * 1000
    throughput = batch_size / (total_time / num_steps)
    
    print(f"Time per step: {time_per_step:.2f} ms")
    print(f"Throughput: {throughput:.2f} samples/sec")

if __name__ == "__main__":
    print("\n--- GPU Benchmarking: Small MLP | Batch Size: 32 ---")
    run_benchmark(128, 64, 32)
    
    print("\n--- GPU Benchmarking: Medium MLP | Batch Size: 32 ---")
    run_benchmark(512, 256, 32)
    
    print("\n--- GPU Benchmarking: Large MLP | Batch Size: 32 ---")
    run_benchmark(1024, 1024, 32)
