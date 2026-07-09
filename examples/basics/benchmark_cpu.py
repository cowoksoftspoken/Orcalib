import orca
import orca.nn as nn
from orca.tensor import Tensor
import numpy as np
import time
import tracemalloc

def create_model(hidden_size):
    return nn.Sequential(
        nn.Linear(784, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 10)
    )

def benchmark(name, model, batch_size, iterations=10):
    print(f"\n--- Benchmarking: {name} | Batch Size: {batch_size} ---")
    loss_fn = nn.CrossEntropyLoss()
    
    # Pre-generate inputs
    x_data = np.random.randn(batch_size, 784).astype(np.float32)
    x = Tensor.from_list(x_data.flatten().tolist(), shape=[batch_size, 784], requires_grad=False)
    
    # Pre-generate one-hot targets
    y_data = np.zeros((batch_size, 10), dtype=np.float32)
    y_data[np.arange(batch_size), np.random.randint(0, 10, batch_size)] = 1.0
    y = Tensor.from_list(y_data.flatten().tolist(), shape=[batch_size, 10], requires_grad=False)
    
    # Warmup
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    
    # Benchmark
    tracemalloc.start()
    start_time = time.time()
    
    for _ in range(iterations):
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        
    end_time = time.time()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    total_time = end_time - start_time
    time_per_iter_ms = (total_time / iterations) * 1000
    ops_per_sec = (batch_size * iterations) / total_time
    peak_mem_mb = peak_mem / (1024 * 1024)
    
    print(f"Time per step: {time_per_iter_ms:.2f} ms")
    print(f"Throughput: {ops_per_sec:.2f} samples/sec")
    print(f"Peak Memory Allocation: {peak_mem_mb:.2f} MB")
    
if __name__ == "__main__":
    configs = [
        ("Small MLP", 64),
        ("Medium MLP", 256),
        ("Large MLP", 512)
    ]
    batch_sizes = [8, 32]
    
    for name, hidden in configs:
        for b_size in batch_sizes:
            model = create_model(hidden)
            benchmark(name, model, b_size)
