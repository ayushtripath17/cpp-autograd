# Autograd exercise

Build a minimal **reverse-mode autograd** system on top of your `Tensor<float>`.

**Starter template:** `include/my_autograd.hpp` (TODOs + suggested fields). This README is the full spec.

Use your existing `learn::Tensor`, `Vector`, and CMake test pattern when you're ready to verify behavior.

---

## Goal

Given a computation graph built from tensor ops, run **forward** to get a loss, then **backward** to populate gradients on every tracked parameter, then **SGD** to update weights.

**Milestone:** train a 2-layer MLP on XOR (or fit `y = mx + b` with one linear layer).

**XOR template:** `tests/test_xor.cpp` — implement `build_xor_data`, `build_model`, `train`, then run `./build/test_xor`.

---

## Suggested order

1. `Variable` + `zero_grad`
2. Forward ops: `add`, `mul`, `matmul`, `relu`
3. `backward` (reverse topological walk)
4. `mse_loss` + backward
5. `SGD` optimizer
6. Wire a tiny training loop

---

## Core type: `Variable` (name yours)

A wrapper around a tensor value that may participate in autograd.

| Function | Purpose |
|----------|---------|
| `Variable(Tensor<float> data, bool requires_grad = false)` | Construct a leaf or param. Leaves used as inputs often set `requires_grad = false`; weights set `true`. |
| `Tensor<float>& data()` / `const Tensor<float>& data() const` | Forward value used in computation. |
| `Tensor<float>& grad()` / `const Tensor<float>& grad() const` | Accumulated ∂loss/∂this. Zero-initialized; same shape as `data` for ops you implement. |
| `bool requires_grad() const` | If false, skip graph building and backward through this node. |
| `void zero_grad()` | Set `grad` to all zeros (same shape as before). |

You also need some way to record **parents** and a **backward callback** per node — design is up to you.

---

## Graph / backward engine

| Function | Purpose |
|----------|---------|
| `void backward(Variable& loss)` | Start reverse-mode AD from `loss`. Set `loss.grad` to a tensor of ones (same shape as `loss.data`, or scalar `1` if loss is a 0-d / 1-element tensor). Walk graph in **reverse** order; each node applies its backward rule and **adds** into each parent's `grad`. Skip nodes with `requires_grad == false`. |
| `void zero_grad(Variable& root)` or `zero_grad(std::vector<Variable*>)` | Zero `grad` on a node and/or all parameters in the model. Called once per training step before forward. |

**Rules:**

- Gradients **accumulate** (`parent.grad += ...`). Always zero before each step unless you intentionally want accumulation.
- If a node has multiple parents, each parent receives its own contribution from the chain rule.
- If a parent is used multiple times in forward, it must receive **sum** of gradients from all uses.

---

## Forward ops (each returns a new `Variable` and records graph edges)

Each op runs forward on `data()`, stores links to inputs, and registers how to propagate `grad` backward.

| Function | Forward | Backward (given `out.grad`) |
|----------|---------|------------------------------|
| `add(a, b)` | `out = a + b` element-wise | `a.grad += out.grad`, `b.grad += out.grad` (shapes must match) |
| `mul(a, b)` | `out = a * b` element-wise | `a.grad += out.grad * b`, `b.grad += out.grad * a` (use forward values of `b`, `a`) |
| `matmul(a, b)` | `out = a.matmul(b)` (your batched matmul rules) | Standard matrix calculus: propagate using transpose of forward operands (implement correctly for your last-2-dims convention) |
| `relu(x)` | `out[i] = max(0, x[i])` | `x.grad += out.grad` where `x > 0`, else `0` |

Optional later: `sub`, `div`, `sum`, `reshape` (only if you can define backward correctly).

---

## Loss

| Function | Purpose |
|----------|---------|
| `Variable mse_loss(const Variable& pred, const Variable& target)` | Forward: mean of `(pred - target)^2` (scalar or 1-element tensor). Backward w.r.t. `pred`: `2 * (pred - target) / n` where `n` is number of elements. `target` is usually constant (no grad or `requires_grad = false`). |

---

## Optimizer

| Function | Purpose |
|----------|---------|
| `SGD(std::vector<Variable*> parameters, float lr)` | Store parameter pointers and learning rate. |
| `void step()` | For each param with `requires_grad`: `param.data -= lr * param.grad` (element-wise). |
| `void zero_grad()` | Call `zero_grad` on all registered parameters. |

---

## Layers (after core autograd works)

| Function | Purpose |
|----------|---------|
| `Linear::forward(x)` | `y = matmul(x, W) + b` (define layout: e.g. `x` is `(batch, in)`, `W` is `(in, out)`). `W`, `b` are `Variable` members with `requires_grad = true`. |
| `Linear::parameters()` | Return `{&W, &b}` for the optimizer. |
| `ReLU::forward(x)` | Apply `relu(x)`. |

| Function | Purpose |
|----------|---------|
| `Sequential::add(layer)` | Own layers in order. |
| `Sequential::forward(x)` | Chain layer forwards. |
| `Sequential::parameters()` | Concatenate all layer parameters. |

---

## Training loop (your `main` or test)

Pseudocode — you implement:

```text
model = ...
optimizer = SGD(model.parameters(), lr)

for epoch in ...:
    optimizer.zero_grad()
    pred = model.forward(x)
    loss = mse_loss(pred, y)
    backward(loss)
    optimizer.step()
```

---

## What to verify

| Check | Expected |
|-------|----------|
| **Grad check** | For small random tensors, compare autograd grad to numeric finite differences on `add`, `mul`, `matmul`, `relu`. |
| **Add** | If `c = a + b` and `loss = sum(c)`, then `a.grad == b.grad == ones`. |
| **XOR** | 2-layer MLP (2→4→1 or similar) reaches low loss after many steps. |

---

## Out of scope (for now)

- GPU
- Adam, conv, batch norm
- Python bindings
- Higher-order grads
- In-place ops that alias storage

---

## Build hint

When you add tests, mirror the existing pattern:

```bash
cmake -S . -B build && cmake --build build && ./build/test_autograd
```

Put headers in `include/` (e.g. `include/autograd/variable.hpp`) and tests in `tests/test_autograd.cpp`; register the target in `CMakeLists.txt`.
