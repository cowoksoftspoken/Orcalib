import orca
import orca.nn as nn
from orca.tensor import Tensor
import numpy as np
import time

def create_model(hidden_size):
    return nn.Sequential(
        nn.Linear(784, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 10)
    )

def benchmark(name, hidden_size, batch_size, device_name, iterations=10):
    model = create_model(hidden_size)
    model.to(device_name)
    loss_fn = nn.CrossEntropyLoss()
    
    # Pre-generate inputs
    x_data = np.random.randn(batch_size, 784).astype(np.float32)
    x = Tensor.from_list(x_data.flatten().tolist(), shape=[batch_size, 784], requires_grad=False).to(device_name)
    
    # Pre-generate one-hot targets
    y_data = np.zeros((batch_size, 10), dtype=np.float32)
    y_data[np.arange(batch_size), np.random.randint(0, 10, batch_size)] = 1.0
    y = Tensor.from_list(y_data.flatten().tolist(), shape=[batch_size, 10], requires_grad=False).to(device_name)
    
    # Warmup
    for _ in range(3):
        list(model.parameters())[0].tensor.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
    
    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        list(model.parameters())[0].tensor.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_iter_ms = (total_time / iterations) * 1000
    ops_per_sec = (batch_size * iterations) / total_time
    
    print(f"[{device_name.upper()}] Time per step: {time_per_iter_ms:.2f} ms | Throughput: {ops_per_sec:.2f} samples/sec")

if __name__ == "__main__":
    configs = [
        ("Small MLP (64)", 64),
        ("Medium MLP (256)", 256),
        ("Large MLP (512)", 512)
    ]
    batch_sizes = [8, 32]
    
    for name, hidden in configs:
        print(f"\n==================================================")
        print(f"BENCHMARKING {name} (3-layer MLP)")
        print(f"==================================================")
        for b_size in batch_sizes:
            print(f"\n--- Batch Size: {b_size} ---")
            benchmark(name, hidden, b_size, "cpu")
            benchmark(name, hidden, b_size, "gpu")
