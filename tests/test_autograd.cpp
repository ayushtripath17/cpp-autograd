#include "my_autograd.hpp"

#include <cmath>
#include <iostream>
#include <memory>
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

using VarPtr = std::shared_ptr<learn::Variable>;

void test_variable_leaf() {
    learn::Variable v(learn::TensorF({2, 2}, 3.f), /*requires_grad=*/true);
    CHECK(v.data().shape()[0] == 2);
    CHECK(v.data().at({0, 0}) == 3.f);
    CHECK(v.requires_grad());
    CHECK(v.grad().size() == v.data().size());
    CHECK(v.grad().at({0, 0}) == 0.f);
}

void test_zero_grad() {
    learn::Variable v(learn::TensorF({3}, 1.f), true);
    v.grad().at({0}) = 5.f;
    v.grad().at({2}) = -2.f;
    v.zero_grad();
    CHECK(v.grad().at({0}) == 0.f);
    CHECK(v.grad().at({2}) == 0.f);
}

void test_add_backward() {
    learn::Variable a(learn::TensorF({2}, {1.f, 2.f}), true);
    learn::Variable b(learn::TensorF({2}, {3.f, 4.f}), true);
    VarPtr c = learn::add(a, b);
    learn::backward(*c);
    CHECK_NEAR(a.grad().at({0}), 1.f, 1e-5f);
    CHECK_NEAR(a.grad().at({1}), 1.f, 1e-5f);
    CHECK_NEAR(b.grad().at({0}), 1.f, 1e-5f);
    CHECK_NEAR(b.grad().at({1}), 1.f, 1e-5f);
}

void test_mul_backward() {
    learn::Variable a(learn::TensorF({2}, {2.f, 3.f}), true);
    learn::Variable b(learn::TensorF({2}, {4.f, 5.f}), true);
    VarPtr c = learn::mul(a, b);
    learn::backward(*c);
    CHECK_NEAR(a.grad().at({0}), 4.f, 1e-5f);
    CHECK_NEAR(a.grad().at({1}), 5.f, 1e-5f);
    CHECK_NEAR(b.grad().at({0}), 2.f, 1e-5f);
    CHECK_NEAR(b.grad().at({1}), 3.f, 1e-5f);
}

void test_relu_backward() {
    learn::Variable x(learn::TensorF({3}, {-1.f, 0.f, 2.f}), true);
    VarPtr y = learn::relu(x);
    learn::backward(*y);
    CHECK_NEAR(x.grad().at({0}), 0.f, 1e-5f);
    CHECK_NEAR(x.grad().at({1}), 0.f, 1e-5f);
    CHECK_NEAR(x.grad().at({2}), 1.f, 1e-5f);
}

void test_matmul_2d_backward() {
    learn::Variable a(learn::TensorF({2, 2}, {1.f, 2.f, 3.f, 4.f}), true);
    learn::Variable b(learn::TensorF({2, 2}, {5.f, 6.f, 7.f, 8.f}), true);
    VarPtr c = learn::matmul(a, b);
    learn::backward(*c);
    CHECK(a.grad().size() == a.data().size());
    CHECK(b.grad().size() == b.data().size());
}

void test_mse_loss() {
    learn::Variable pred(learn::TensorF({2}, {1.f, 2.f}), true);
    learn::Variable target(learn::TensorF({2}, {0.f, 0.f}), false);
    VarPtr loss = learn::mse_loss(pred, target);
    learn::backward(*loss);
    CHECK_NEAR(pred.grad().at({0}), 1.f, 1e-5f);
    CHECK_NEAR(pred.grad().at({1}), 2.f, 1e-5f);
}

void test_sgd_step() {
    learn::Variable w(learn::TensorF({2}, {1.f, 1.f}), true);
    w.grad().at({0}) = 2.f;
    w.grad().at({1}) = 4.f;
    learn::SGD opt({&w}, 0.1f);
    opt.step();
    CHECK_NEAR(w.data().at({0}), 1.f - 0.1f * 2.f, 1e-5f);
    CHECK_NEAR(w.data().at({1}), 1.f - 0.1f * 4.f, 1e-5f);
}

}  // namespace

int main() {
    run_test("variable_leaf", test_variable_leaf);
    run_test("zero_grad", test_zero_grad);
    run_test("add_backward", test_add_backward);
    run_test("mul_backward", test_mul_backward);
    run_test("relu_backward", test_relu_backward);
    run_test("matmul_2d_backward", test_matmul_2d_backward);
    run_test("mse_loss", test_mse_loss);
    run_test("sgd_step", test_sgd_step);

    std::cout << tests_run << " checks, " << tests_failed << " failed\n";
    return tests_failed == 0 ? 0 : 1;
}
