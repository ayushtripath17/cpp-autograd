# cpp-autograd

A from-scratch deep learning stack in modern C++ — no external ML libraries, no PyTorch, no Eigen. Built layer by layer: dynamic containers → matrices → tensors → autograd → neural network training.

## What this is

Each stage implements core data structures and algorithms by hand, with unit tests at every step. The project trains a small MLP on the XOR dataset using reverse-mode automatic differentiation and SGD.

```
Vector  →  Matrix  →  Tensor  →  Autograd  →  XOR training
```

## Features

- **`Vector<T>`** — RAII dynamic array (Rule of 5, `push_back`, `reserve`, iterators)
- **`Matrix<T>`** — row-major 2D storage with element access and basic ops
- **`Tensor<T>`** — N-dimensional row-major tensors with shape, strides, views, transpose, reshape, and batched matmul
- **Autograd** — `Variable` computation graph, forward ops (`add`, `mul`, `matmul`, `relu`, `broadcast_add`), `mse_loss`, reverse-mode `backward`, and `SGD`
- **Layers** — `Linear`, `ReLU`, `Sequential`
- **XOR demo** — 2-layer MLP (`Linear(2,4) → ReLU → Linear(4,1)`) trained on the classic non-linearly-separable dataset

## Requirements

- C++17 compiler (Clang or GCC)
- CMake 3.16+

## Build & test

```bash
cmake -S . -B build
cmake --build build
```

Run all tests:

```bash
./build/test_vector
./build/test_matrix
./build/test_tensor
./build/test_autograd
./build/test_xor
```

Or rebuild and run in one step:

```bash
cmake -S . -B build && cmake --build build && ./build/test_autograd && ./build/test_xor
```

## Project layout

```
include/
  my_vector.hpp    # Dynamic array
  my_matrix.hpp    # 2D matrix
  my_tensor.hpp    # N-D tensor + views + matmul
  my_autograd.hpp  # Variable, ops, backward, SGD, layers

tests/
  test_vector.cpp
  test_matrix.cpp
  test_tensor.cpp
  test_autograd.cpp
  test_xor.cpp     # End-to-end XOR training
```

## Tensor layout convention

Batch-first, row-major (NumPy-style):

| Tensor | Shape | Role |
|--------|-------|------|
| `x` | `(batch, in_features)` | Input |
| `W` | `(in_features, out_features)` | Weight |
| `b` | `(out_features,)` | Bias (broadcast over batch) |
| output | `(batch, out_features)` | `matmul(x, W) + b` |

## Autograd overview

1. **Forward** — ops create new `Variable` nodes, record parent pointers, and attach a `backward_fn` lambda.
2. **Loss** — `mse_loss(pred, target)` returns a scalar loss node.
3. **Backward** — topological walk from the loss; each node accumulates gradients into its parents.
4. **Optimize** — `SGD` updates parameter `data` using accumulated `grad`.

Training step:

```cpp
optimizer.zero_grad();
auto nodes = model.forward(x);
auto& pred = *nodes.back();
auto loss = mse_loss(pred, y);
backward(*loss);
optimizer.step();
```

Keep graph nodes alive (via `shared_ptr`) until `backward` finishes — parent pointers are non-owning.

## XOR training

The XOR dataset is 4 samples of 2-bit inputs with targets `[0, 1, 1, 0]`. A single linear layer cannot solve XOR; the hidden ReLU layer learns a non-linear decision boundary.

Training can be sensitive to random weight initialization. For reproducible runs, use a fixed seed in `Linear` rather than `std::random_device`.

## Out of scope

- GPU / CUDA
- Adam, convolutions, batch norm
- Python bindings
- Second-order gradients
