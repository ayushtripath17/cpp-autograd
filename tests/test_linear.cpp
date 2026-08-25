// cmake -S . -B build && cmake --build build --target test_linear && ./build/test_linear

#include "my_autograd.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

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

bool is_finite_tensor(const learn::TensorF& t) {
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (!std::isfinite(t.data()[i])) {
            return false;
        }
    }
    return true;
}

bool tensors_equal(const learn::TensorF& a, const learn::TensorF& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a.data()[i] != b.data()[i]) {
            return false;
        }
    }
    return true;
}

bool params_equal(const std::vector<learn::Variable*>& a,
                  const std::vector<learn::Variable*>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!tensors_equal(a[i]->data(), b[i]->data())) {
            return false;
        }
    }
    return true;
}

struct LinearDataset {
    learn::Variable x;
    learn::Variable y;
};

// x in [0, 1), y = 2x + 1  →  target weight ≈ 2, bias ≈ 1
// (scaled features keep SGD stable at a reasonable learning rate)
LinearDataset build_linear_data() {
    learn::Variable x(learn::TensorF({100, 1}), /*requires_grad=*/false);
    learn::Variable y(learn::TensorF({100, 1}), /*requires_grad=*/false);

    for (std::size_t i = 0; i < 100; ++i) {
        const float xi = static_cast<float>(i) / 100.f;
        x.data().at({i, 0}) = xi;
        y.data().at({i, 0}) = 2.f * xi + 1.f;
    }

    return LinearDataset{x, y};
}

learn::Sequential build_linear_model(std::uint32_t seed) {
    std::mt19937 rng(seed);
    learn::Sequential model;
    model.add(learn::Linear(1, 1, rng));
    return model;
}

float compute_loss(learn::Sequential& model,
                   learn::Variable& x,
                   learn::Variable& y) {
    auto pred = model.forward(x);
    return learn::mse_loss(pred, y)->data().at({0});
}

float train(learn::Sequential& model,
            learn::Variable& x,
            learn::Variable& y,
            float lr,
            int epochs) {
    learn::SGD opt(model.parameters(), lr);

    for (int i = 0; i < epochs; ++i) {
        opt.zero_grad();
        std::shared_ptr<learn::Variable> pred = model.forward(x);
        auto loss = mse_loss(pred, y);
        learn::backward(*loss);
        opt.step();
    }

    return compute_loss(model, x, y);
}

void test_final_loss_lower_than_initial() {
    LinearDataset data = build_linear_data();
    learn::Sequential model = build_linear_model(42);

    const float initial_loss = compute_loss(model, data.x, data.y);
    const float final_loss = train(model, data.x, data.y, 0.05f, 5000);

    CHECK(std::isfinite(initial_loss));
    CHECK(std::isfinite(final_loss));
    CHECK(final_loss < 0.05f);
    CHECK(final_loss < 0.1f * initial_loss);
}

void test_learned_weight_and_bias() {
    LinearDataset data = build_linear_data();
    learn::Sequential model = build_linear_model(42);
    (void)train(model, data.x, data.y, 0.05f, 5000);

    auto params = model.parameters();
    CHECK(params.size() == 2);
    // Linear::parameters() returns {W, b}
    const float w = params[0]->data().at({0, 0});
    const float b = params[1]->data().at({0});
    CHECK_NEAR(w, 2.f, 0.1f);
    CHECK_NEAR(b, 1.f, 0.1f);
}

void test_no_nan_or_inf() {
    LinearDataset data = build_linear_data();
    learn::Sequential model = build_linear_model(42);
    learn::SGD opt(model.parameters(), 0.05f);

    for (int i = 0; i < 200; ++i) {
        opt.zero_grad();
        auto pred = model.forward(data.x);
        auto loss = mse_loss(pred, data.y);
        CHECK(std::isfinite(loss->data().at({0})));
        CHECK(is_finite_tensor(pred->data()));
        learn::backward(*loss);
        opt.step();

        for (learn::Variable* p : model.parameters()) {
            CHECK(is_finite_tensor(p->data()));
            CHECK(is_finite_tensor(p->grad()));
        }
    }

    const float final_loss = compute_loss(model, data.x, data.y);
    CHECK(std::isfinite(final_loss));
}

void test_zero_grad_prevents_accumulation() {
    LinearDataset data = build_linear_data();
    learn::Sequential model = build_linear_model(7);
    auto params = model.parameters();

    auto run_backward = [&]() {
        auto pred = model.forward(data.x);
        auto loss = mse_loss(pred, data.y);
        learn::backward(*loss);
    };

    // One backward — reference gradient on W.
    for (learn::Variable* p : params) {
        p->zero_grad();
    }
    run_backward();
    const float g_once = params[0]->grad().at({0, 0});

    // Second backward without zero_grad — grads should accumulate (~2x).
    run_backward();
    const float g_accum = params[0]->grad().at({0, 0});
    CHECK(std::fabs(g_accum) > std::fabs(g_once) * 1.5f);

    // zero_grad then one backward — back to single-step magnitude.
    for (learn::Variable* p : params) {
        p->zero_grad();
    }
    CHECK_NEAR(params[0]->grad().at({0, 0}), 0.f, 1e-7f);
    run_backward();
    CHECK_NEAR(params[0]->grad().at({0, 0}), g_once, 1e-4f);
}

void test_same_seed_reproducible() {
    LinearDataset data = build_linear_data();
    const std::uint32_t seed = 42;
    const float lr = 0.05f;
    const int epochs = 2000;

    learn::Sequential m1 = build_linear_model(seed);
    const float init1 = compute_loss(m1, data.x, data.y);
    const float final1 = train(m1, data.x, data.y, lr, epochs);

    learn::Sequential m2 = build_linear_model(seed);
    const float init2 = compute_loss(m2, data.x, data.y);
    const float final2 = train(m2, data.x, data.y, lr, epochs);

    CHECK(init1 == init2);
    CHECK(final1 == final2);
    CHECK(params_equal(m1.parameters(), m2.parameters()));
}

void test_graph_nodes_expire_each_iteration() {
    LinearDataset data = build_linear_data();
    learn::Sequential model = build_linear_model(42);
    learn::SGD opt(model.parameters(), 0.05f);

    for (int i = 0; i < 5; ++i) {
        std::weak_ptr<learn::Variable> weak_pred;
        std::weak_ptr<learn::Variable> weak_loss;
        {
            opt.zero_grad();
            auto pred = model.forward(data.x);
            auto loss = mse_loss(pred, data.y);
            weak_pred = pred;
            weak_loss = loss;
            CHECK(!weak_pred.expired());
            CHECK(!weak_loss.expired());
            learn::backward(*loss);
            opt.step();
        }
        // Intermediate graph nodes must not leak across iterations.
        CHECK(weak_pred.expired());
        CHECK(weak_loss.expired());
    }
}

void test_xor_fixed_seed_still_passes() {
    // Smoke check: fixed-seed XOR MLP still converges (regression vs linear work).
    learn::Variable x(learn::TensorF({4, 2}), false);
    learn::Variable y(learn::TensorF({4, 1}), false);
    x.data().at({0, 0}) = 0; x.data().at({0, 1}) = 0;
    x.data().at({1, 0}) = 0; x.data().at({1, 1}) = 1;
    x.data().at({2, 0}) = 1; x.data().at({2, 1}) = 0;
    x.data().at({3, 0}) = 1; x.data().at({3, 1}) = 1;
    y.data().at({0, 0}) = 0;
    y.data().at({1, 0}) = 1;
    y.data().at({2, 0}) = 1;
    y.data().at({3, 0}) = 0;

    std::mt19937 rng(42);
    learn::Sequential model;
    model.add(learn::Linear(2, 4, rng));
    model.add(learn::ReLU());
    model.add(learn::Linear(4, 1, rng));

    const float final_loss = train(model, x, y, 0.05f, 10000);
    CHECK(final_loss < 0.05f);

    auto pred = model.forward(x);
    CHECK_NEAR(pred->data().at({0, 0}), 0.f, 0.2f);
    CHECK_NEAR(pred->data().at({1, 0}), 1.f, 0.2f);
    CHECK_NEAR(pred->data().at({2, 0}), 1.f, 0.2f);
    CHECK_NEAR(pred->data().at({3, 0}), 0.f, 0.2f);
}

void test_linear_predictions() {
    LinearDataset data = build_linear_data();
    learn::Sequential model = build_linear_model(42);
    (void)train(model, data.x, data.y, 0.05f, 5000);

    auto pred = model.forward(data.x);
    // y = 2x + 1 with x = i/100
    CHECK_NEAR(pred->data().at({0, 0}), 1.f, 0.15f);
    CHECK_NEAR(pred->data().at({1, 0}), 2.f * 0.01f + 1.f, 0.15f);
    CHECK_NEAR(pred->data().at({50, 0}), 2.f * 0.50f + 1.f, 0.15f);
    CHECK_NEAR(pred->data().at({99, 0}), 2.f * 0.99f + 1.f, 0.15f);
}

}  // namespace

int main() {
    run_test("final_loss_lower_than_initial", test_final_loss_lower_than_initial);
    run_test("learned_weight_and_bias", test_learned_weight_and_bias);
    run_test("no_nan_or_inf", test_no_nan_or_inf);
    run_test("zero_grad_prevents_accumulation", test_zero_grad_prevents_accumulation);
    run_test("same_seed_reproducible", test_same_seed_reproducible);
    run_test("graph_nodes_expire_each_iteration", test_graph_nodes_expire_each_iteration);
    run_test("xor_fixed_seed_still_passes", test_xor_fixed_seed_still_passes);
    run_test("linear_predictions", test_linear_predictions);

    std::cout << tests_run << " checks, " << tests_failed << " failed\n";
    return tests_failed == 0 ? 0 : 1;
}
