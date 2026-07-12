# Orca: Progressive Machine Learning Framework

**"Simple by default. Powerful when needed."**

Orca is a lightweight, modular, and high-performance Machine Learning framework built from the ground up. It leverages the memory safety and native execution speed of **Rust** for its core computational backend, while exposing a clean, intuitive, and PyTorch-compatible API through **Python**. 

Currently at version 1.0.0, Orca focuses on providing an extensible architecture where foundational elements like Autograd engines and mathematical primitives are completely decoupled from the physical execution layer (CPU/GPU).

---

## Core Features

- **PyTorch-like Python Frontend**: Designed for immediate familiarity. The framework implements standard abstractions such as `Tensor`, `nn.Module`, `optim.SGD`, and `DataLoader`.
- **Reverse-Mode Autograd Engine**: A robust, tape-based automatic differentiation engine written entirely in Rust, dynamically building computation graphs during the forward pass.
- **Modular Backend Architecture**: Core ML primitives (mathematical operations, multidimensional shapes, broadcasting) are strictly decoupled from hardware backends. Backends can be swapped seamlessly without rewriting the autograd or frontend layers.
- **Safe, Fast, and SIMD-Ready**: Built 100% in Rust with zero legacy C/C++ dependencies. The CPU backend uses custom aligned memory allocators (64-byte alignment) to ensure type-safe slice casting and future-proof AVX-512/SIMD support.
- **Seamless Python Integration**: Native bindings generated using [PyO3](https://pyo3.rs/) and built using [Maturin](https://maturin.rs/) to guarantee zero-overhead interoperability.

---

## Architecture Structure

The repository is highly decoupled to prevent circular dependencies and enforce clean abstractions. The workspace is divided into the following crates:

- `orca-core/`: The foundational layer. Defines core traits (`Backend`), `Shape`, `DType`, `Device`, and unified error handling (`OrcaError`, `Result`).
- `orca-tensor/`: The multidimensional array representation (`Tensor<B: Backend>`) and forward-pass mathematical operations.
- `orca-autograd/`: The reverse-mode Automatic Differentiation Engine (`Autodiff<B>`) utilizing a Tape-based computation graph for dynamic backpropagation.
- `orca-backend-cpu/`: The reference implementation for a single-threaded CPU Backend featuring robust type dispatching and aligned raw memory storage.
- `orca-backend-gpu/`: An experimental wgpu-based backend designed for cross-platform parallel shader execution.
- `orca-python/`: The Rust-to-Python FFI (Foreign Function Interface) bindings.
- `python/orca/`: The Python frontend providing Object-Oriented ML blocks (`nn`, `optim`, `data`) and autocompletion interfaces.

---

## Installation & Setup

### Prerequisites
- **Python:** 3.10 or higher.
- **Rust:** Stable toolchain via [rustup](https://rustup.rs/).

### Development Installation

1. Clone the repository to your local machine.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   
   # On Linux / macOS:
   source .venv/bin/activate
   
   # On Windows:
   .venv\Scripts\Activate.ps1
   ```
3. Install the Rust bindings compiler (`maturin`) and build the framework:
   ```bash
   pip install maturin
   maturin develop --release
   ```

---

## Quick Start Guide

The Python API is designed to mimic standard deep learning workflows. Below is an example of creating a Multi-Layer Perceptron (MLP) and running a forward and backward pass.

```python
import orca
from orca import Tensor
import orca.nn as nn
import orca.optim as optim

# 1. Define the Model Architecture
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

# 2. Define Loss Function and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 3. Create Dummy Data
dummy_input = orca.randn([32, 64], requires_grad=False)
dummy_target = orca.zeros([32, 10], requires_grad=False) # Labels representation

# 4. Forward Pass
predictions = model(dummy_input)
loss = loss_fn(predictions, dummy_target)

# 5. Backward Pass and Optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Training Step Completed. Loss: {loss.to_list()}")
```

---

## Roadmap and Project Status

The framework is actively evolving. The current state reflects a stable CPU foundation capable of training standard classification models.

- **Phase 1 (Completed):** Core Foundation, Tensor Structs, and Basic CPU primitives.
- **Phase 2 (Completed):** Autograd Engine & PyO3 Bindings implementation.
- **Phase 3 (Completed):** Broadcasting, Non-linear Math Operations, and Logic gates convergence verification.
- **Phase 4 (Completed):** ML Primitives implementation (`nn.Linear`, `CrossEntropyLoss`, `SGD`) and successful MNIST Classification verification.
- **Phase 5 (In Progress):** GPU Acceleration Backend via `wgpu` (Parallel shader execution).
- **Phase 6 (Planned):** Advanced Optimizers (Adam, RMSprop) and broader tensor algebraic operations.

---

## Contributing Guidelines

Contributions, bug reports, and feature requests are welcome. When contributing, please ensure strict adherence to the architectural rules defined in `doc/foundation/`, `doc/foundation/ARCHITECTURE.md`, `doc/foundation/ERROR_HANDLING.md`, `doc/foundation/TRAINING_PIPELINE.md` and `doc/foundation/BENCHMARKS.md`.

> **For AI Agents**: please read all files in `doc/foundation/` and `agents/AGENTS.md` carefully before making any changes.

1. **Error Handling:** Usage of `.unwrap()` or `panic!` is strictly prohibited in library code (`src/`). Proper error propagation (`Result`, `OrcaError`) must be used at all times.
2. **Crate Hierarchy:** Do not introduce circular dependencies. `orca-core` must remain independent, and `orca-autograd` must only depend on `orca-tensor` and `orca-core`.
3. **Continuous Integration:** Ensure all backend implementations compile cleanly without Clippy warnings (`cargo clippy --workspace -- -D warnings`).

## License

This project is licensed under the MIT License or Apache-2.0 License.
