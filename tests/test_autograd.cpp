#include "my_autograd.hpp"

#include <cmath>
#include <functional>
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
using LossFn = std::function<VarPtr()>;

// Compare autograd gradients to central finite differences for one Variable.
void grad_check(const VarPtr& v, const LossFn& make_loss,
                float eps = 1e-4f, float rel_tol = 1e-2f) {
    for (std::size_t i = 0; i < v->data().size(); ++i) {
        const float orig = v->data().data()[i];

        v->data().data()[i] = orig + eps;
        const float loss_plus = make_loss()->data().data()[0];

        v->data().data()[i] = orig - eps;
        const float loss_minus = make_loss()->data().data()[0];

        v->data().data()[i] = orig;

        const float numerical = (loss_plus - loss_minus) / (2.f * eps);

        v->zero_grad();
        VarPtr loss = make_loss();
        learn::backward(*loss);
        const float analytical = v->grad().data()[i];

        const float scale = std::max({1.f, std::fabs(analytical), std::fabs(numerical)});
        CHECK(std::fabs(analytical - numerical) / scale < rel_tol);
    }
}

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
    VarPtr a = std::make_shared<learn::Variable>(learn::TensorF({2}, {1.f, 2.f}), true);
    VarPtr b = std::make_shared<learn::Variable>(learn::TensorF({2}, {3.f, 4.f}), true);
    VarPtr c = learn::add(a, b);
    learn::backward(*c);
    CHECK_NEAR(a->grad().at({0}), 1.f, 1e-5f);
    CHECK_NEAR(a->grad().at({1}), 1.f, 1e-5f);
    CHECK_NEAR(b->grad().at({0}), 1.f, 1e-5f);
    CHECK_NEAR(b->grad().at({1}), 1.f, 1e-5f);
}

void test_mul_backward() {
    VarPtr a = std::make_shared<learn::Variable>(learn::TensorF({2}, {2.f, 3.f}), true);
    VarPtr b = std::make_shared<learn::Variable>(learn::TensorF({2}, {4.f, 5.f}), true);
    VarPtr c = learn::mul(a, b);
    learn::backward(*c);
    CHECK_NEAR(a->grad().at({0}), 4.f, 1e-5f);
    CHECK_NEAR(a->grad().at({1}), 5.f, 1e-5f);
    CHECK_NEAR(b->grad().at({0}), 2.f, 1e-5f);
    CHECK_NEAR(b->grad().at({1}), 3.f, 1e-5f);
}

void test_relu_backward() {
    VarPtr x = std::make_shared<learn::Variable>(learn::TensorF({3}, {-1.f, 0.f, 2.f}), true);
    VarPtr y = learn::relu(x);
    learn::backward(*y);
    CHECK_NEAR(x->grad().at({0}), 0.f, 1e-5f);
    CHECK_NEAR(x->grad().at({1}), 0.f, 1e-5f);
    CHECK_NEAR(x->grad().at({2}), 1.f, 1e-5f);
}

void test_matmul_2d_backward() {
    VarPtr a = std::make_shared<learn::Variable>(learn::TensorF({2, 2}, {1.f, 2.f, 3.f, 4.f}), true);
    VarPtr b = std::make_shared<learn::Variable>(learn::TensorF({2, 2}, {5.f, 6.f, 7.f, 8.f}), true);
    VarPtr c = learn::matmul(a, b);
    learn::backward(*c);
    CHECK(a->grad().size() == a->data().size());
    CHECK(b->grad().size() == b->data().size());
}

void test_mse_loss() {
    VarPtr pred = std::make_shared<learn::Variable>(learn::TensorF({2}, {1.f, 2.f}), true);
    learn::Variable target(learn::TensorF({2}, {0.f, 0.f}), false);
    VarPtr loss = learn::mse_loss(pred, target);
    learn::backward(*loss);
    CHECK_NEAR(pred->grad().at({0}), 1.f, 1e-5f);
    CHECK_NEAR(pred->grad().at({1}), 2.f, 1e-5f);
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

void test_sub_forward() {
    VarPtr a = std::make_shared<learn::Variable>(learn::TensorF({2}, {5.f, 7.f}), true);
    VarPtr b = std::make_shared<learn::Variable>(learn::TensorF({2}, {2.f, 3.f}), true);
    VarPtr c = learn::sub(a, b);
    CHECK_NEAR(c->data().at({0}), 3.f, 1e-5f);
    CHECK_NEAR(c->data().at({1}), 4.f, 1e-5f);
}

void test_sub_backward() {
    VarPtr a = std::make_shared<learn::Variable>(learn::TensorF({2}, {5.f, 7.f}), true);
    VarPtr b = std::make_shared<learn::Variable>(learn::TensorF({2}, {2.f, 3.f}), true);
    VarPtr c = learn::sub(a, b);
    learn::backward(*c);
    CHECK_NEAR(a->grad().at({0}), 1.f, 1e-5f);
    CHECK_NEAR(a->grad().at({1}), 1.f, 1e-5f);
    CHECK_NEAR(b->grad().at({0}), -1.f, 1e-5f);
    CHECK_NEAR(b->grad().at({1}), -1.f, 1e-5f);
}

void test_neg_forward() {
    VarPtr x = std::make_shared<learn::Variable>(learn::TensorF({3}, {2.f, -3.f, 0.f}), true);
    VarPtr y = learn::neg(x);
    CHECK_NEAR(y->data().at({0}), -2.f, 1e-5f);
    CHECK_NEAR(y->data().at({1}), 3.f, 1e-5f);
    CHECK_NEAR(y->data().at({2}), 0.f, 1e-5f);
}

void test_neg_backward() {
    VarPtr x = std::make_shared<learn::Variable>(learn::TensorF({2}, {4.f, -1.f}), true);
    VarPtr y = learn::neg(x);
    learn::backward(*y);
    CHECK_NEAR(x->grad().at({0}), -1.f, 1e-5f);
    CHECK_NEAR(x->grad().at({1}), -1.f, 1e-5f);
}

void test_sum_forward() {
    VarPtr x = std::make_shared<learn::Variable>(learn::TensorF({3}, {1.f, 2.f, 3.f}), true);
    VarPtr total = learn::sum(x);
    CHECK(total->data().size() == 1);
    CHECK_NEAR(total->data().at({0}), 6.f, 1e-5f);
}

void test_sum_backward() {
    VarPtr x = std::make_shared<learn::Variable>(learn::TensorF({3}, {1.f, 2.f, 3.f}), true);
    VarPtr total = learn::sum(x);
    learn::backward(*total);
    CHECK_NEAR(x->grad().at({0}), 1.f, 1e-5f);
    CHECK_NEAR(x->grad().at({1}), 1.f, 1e-5f);
    CHECK_NEAR(x->grad().at({2}), 1.f, 1e-5f);
}

void test_sum_backward_2d() {
    VarPtr x = std::make_shared<learn::Variable>(learn::TensorF({2, 2}, {1.f, 2.f, 3.f, 4.f}), true);
    VarPtr total = learn::sum(x);
    learn::backward(*total);
    CHECK_NEAR(x->grad().at({0, 0}), 1.f, 1e-5f);
    CHECK_NEAR(x->grad().at({0, 1}), 1.f, 1e-5f);
    CHECK_NEAR(x->grad().at({1, 0}), 1.f, 1e-5f);
    CHECK_NEAR(x->grad().at({1, 1}), 1.f, 1e-5f);
}

void test_grad_check_add() {
    VarPtr a = std::make_shared<learn::Variable>(learn::TensorF({2}, {1.5f, -0.7f}), true);
    VarPtr b = std::make_shared<learn::Variable>(learn::TensorF({2}, {2.1f, 0.3f}), true);
    const LossFn make_loss = [&]() {
        VarPtr c = learn::add(a, b);
        return learn::sum(c);
    };
    grad_check(a, make_loss);
    grad_check(b, make_loss);
}

void test_grad_check_sub() {
    VarPtr a = std::make_shared<learn::Variable>(learn::TensorF({2}, {3.2f, 1.1f}), true);
    VarPtr b = std::make_shared<learn::Variable>(learn::TensorF({2}, {0.5f, 2.0f}), true);
    const LossFn make_loss = [&]() {
        VarPtr c = learn::sub(a, b);
        return learn::sum(c);
    };
    grad_check(a, make_loss);
    grad_check(b, make_loss);
}

void test_grad_check_mul() {
    VarPtr a = std::make_shared<learn::Variable>(learn::TensorF({2}, {1.2f, -0.8f}), true);
    VarPtr b = std::make_shared<learn::Variable>(learn::TensorF({2}, {0.5f, 1.5f}), true);
    const LossFn make_loss = [&]() {
        VarPtr c = learn::mul(a, b);
        return learn::sum(c);
    };
    grad_check(a, make_loss);
    grad_check(b, make_loss);
}

void test_grad_check_neg() {
    VarPtr x = std::make_shared<learn::Variable>(learn::TensorF({3}, {1.0f, -2.0f, 0.5f}), true);
    const LossFn make_loss = [&]() {
        VarPtr y = learn::neg(x);
        return learn::sum(y);
    };
    grad_check(x, make_loss);
}

void test_grad_check_matmul() {
    VarPtr a = std::make_shared<learn::Variable>(learn::TensorF({2, 2}, {0.3f, -0.5f, 1.1f, 0.7f}), true);
    VarPtr b = std::make_shared<learn::Variable>(learn::TensorF({2, 2}, {0.2f, 0.4f, -0.6f, 0.8f}), true);
    const LossFn make_loss = [&]() {
        VarPtr c = learn::matmul(a, b);
        return learn::sum(c);
    };
    grad_check(a, make_loss);
    grad_check(b, make_loss);
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
    run_test("sub_forward", test_sub_forward);
    run_test("sub_backward", test_sub_backward);
    run_test("neg_forward", test_neg_forward);
    run_test("neg_backward", test_neg_backward);
    run_test("sum_forward", test_sum_forward);
    run_test("sum_backward", test_sum_backward);
    run_test("sum_backward_2d", test_sum_backward_2d);
    run_test("grad_check_add", test_grad_check_add);
    run_test("grad_check_sub", test_grad_check_sub);
    run_test("grad_check_mul", test_grad_check_mul);
    run_test("grad_check_neg", test_grad_check_neg);
    run_test("grad_check_matmul", test_grad_check_matmul);

    std::cout << tests_run << " checks, " << tests_failed << " failed\n";
    return tests_failed == 0 ? 0 : 1;
}
