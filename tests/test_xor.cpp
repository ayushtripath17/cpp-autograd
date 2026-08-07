// XOR training exercise — implement the TODOs below.
//
// Goal: train a 2-layer MLP on the XOR dataset using your autograd stack.
//
// Model (suggested):
//   Sequential: Linear(2, 4) → ReLU → Linear(4, 1)
//
// Data layout (match autograd convention):
//   x: (batch, 2)   — requires_grad = false
//   y: (batch, 1)   — requires_grad = false
//
// Training step:
//   optimizer.zero_grad()
//   pred = model.forward(x)
//   loss = mse_loss(pred, y)
//   backward(loss)
//   optimizer.step()
//
// Suggested order:
//   1. build_xor_data() — fill x and y Variables
//   2. build_model() — Sequential with two Linear layers and ReLU
//   3. train() — loop epochs, return final loss scalar
//   4. test_xor_learns — assert loss < threshold after enough epochs
//
// Hints:
//   - lr around 0.01–0.1; try 500–5000 epochs
//   - print loss every N epochs while debugging
//   - final loss < 0.05 (or predictions within 0.15 of targets) is a reasonable pass
//
// Build & run:
//   cmake -S . -B build && cmake --build build && ./build/test_xor

#include "my_autograd.hpp"

#include <cmath>
#include <iostream>
#include <utility>

namespace {

int tests_run = 0;
int tests_failed = 0;

void run_test(const char* name, void (*test)()) {
    const int failed_before = tests_failed;
    std::cout << "RUN: " << name << std::endl;
    test();
    if (tests_failed == failed_before) {
        std::cout << "PASS: " << name << std::endl;
    } else {
        std::cerr << "FAIL (checks): " << name << std::endl;
    }
}

#define CHECK(condition)                                                     \
    do {                                                                     \
        ++tests_run;                                                         \
        if (!(condition)) {                                                  \
            std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__ << " — "    \
                      << #condition << '\n';                                 \
            ++tests_failed;                                                  \
        }                                                                    \
    } while (0)

#define CHECK_NEAR(a, b, eps) CHECK(std::fabs((a) - (b)) < (eps))

// ---------------------------------------------------------------------------
// TODO 1: XOR dataset as Variables
// ---------------------------------------------------------------------------
// Return {x, y} where:
//   x shape (4, 2):  [0,0] [0,1] [1,0] [1,1]
//   y shape (4, 1):  0, 1, 1, 0
// Both should have requires_grad = false.
struct XORDataset {
    learn::Variable x;
    learn::Variable y;
};

XORDataset build_xor_data() {
    // TODO: construct x and y from the values above.
    learn::Variable x(learn::TensorF({4, 2}), /*requires_grad=*/false);
    learn::Variable y(learn::TensorF({4, 1}), /*requires_grad=*/false);
    
    x.data().at({0, 0}) = 0;
    x.data().at({0, 1}) = 0;
    x.data().at({1, 0}) = 0;
    x.data().at({1, 1}) = 1;
    x.data().at({2, 0}) = 1;
    x.data().at({2, 1}) = 0;
    x.data().at({3, 0}) = 1;
    x.data().at({3, 1}) = 1;

    y.data().at({0, 0}) = 0;
    y.data().at({1, 0}) = 1;
    y.data().at({2, 0}) = 1;
    y.data().at({3, 0}) = 0;

    return XORDataset{x, y};
}

// ---------------------------------------------------------------------------
// TODO 2: model
// ---------------------------------------------------------------------------
learn::Sequential build_model() {
    // TODO: Sequential with Linear(2, 4), ReLU, Linear(4, 1).
    learn::Sequential model;
    model.add(learn::Linear(2, 4));
    model.add(learn::ReLU());
    model.add(learn::Linear(4, 1));

    return model;
}

// ---------------------------------------------------------------------------
// TODO 3: training loop
// ---------------------------------------------------------------------------
float train(learn::Sequential& model,
            learn::Variable& x,
            learn::Variable& y,
            float lr,
            int epochs) {
    
    learn::SGD opt = learn::SGD(model.parameters(), lr);

    for (int i = 0; i < epochs; i++) {
        opt.zero_grad();

        std::shared_ptr<learn::Variable> pred = model.forward(x);
        auto loss = mse_loss(pred, y);

        // std::cout << loss->data().at({0}) << std::endl;
        learn::backward(*loss);
        opt.step();
    }
    std::shared_ptr<learn::Variable> final_pred = model.forward(x);
    return learn::mse_loss(final_pred, y)->data().at({0});  // stub — replace
}

// ---------------------------------------------------------------------------
// TODO 4: end-to-end test
// ---------------------------------------------------------------------------
void test_xor_learns() {
    XORDataset data = build_xor_data();
    learn::Sequential model = build_model();

    const float lr = 0.05f;
    const int epochs = 10000;

    const float final_loss = train(model, data.x, data.y, lr, epochs);

    std::cout << final_loss << "\n";
    // TODO: tighten once training works (e.g. final_loss < 0.05f)
    CHECK(final_loss < 0.05f);
}

void test_xor_predictions() {
    XORDataset data = build_xor_data();
    learn::Sequential model = build_model();

    (void)train(model, data.x, data.y, 0.05f, 10000);

    // TODO: forward once more, check each prediction is close to target.
    // Example (row-major (4,1) output):
    //   CHECK_NEAR(pred.at({0, 0}), 0.f, 0.2f);
    //   CHECK_NEAR(pred.at({1, 0}), 1.f, 0.2f);
    //   ...
    std::shared_ptr<learn::Variable> test_pred = model.forward(data.x);
    learn::Tensor<float> pred = test_pred->data();
    CHECK_NEAR(pred.at({0, 0}), 0.f, 0.2f);
    CHECK_NEAR(pred.at({1, 0}), 1.f, 0.2f);
    CHECK_NEAR(pred.at({2, 0}), 1.f, 0.2f);
    CHECK_NEAR(pred.at({3, 0}), 0.f, 0.2f);
}

}  // namespace

int main() {
    run_test("xor_learns", test_xor_learns);
    run_test("xor_predictions", test_xor_predictions);

    std::cout << tests_run << " checks, " << tests_failed << " failed\n";
    return tests_failed == 0 ? 0 : 1;
}
