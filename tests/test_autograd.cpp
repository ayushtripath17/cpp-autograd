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

void test_graph_lifetime() {
    std::weak_ptr<learn::Variable> observed;
    {
        auto a = std::make_shared<learn::Variable>(learn::TensorF({2, 2}, {0.3f, -0.5f, 1.1f, 0.7f}), true);
        auto b = std::make_shared<learn::Variable>(learn::TensorF({2, 2}, {0.2f, 0.4f, -0.6f, 0.8f}), true);
        auto out = add(a, b);
        observed = out;

        CHECK(!observed.expired());
    }
    CHECK(observed.expired());
}

template <typename Fn>
bool throws_exception(Fn&& fn) {
    try {
        fn();
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

// Adam update helper matching learn::Adam::step for one scalar element.
float adam_expected(float theta, float g, float m, float v,
                    float lr, float beta1, float beta2, float eps, int t,
                    float* m_out = nullptr, float* v_out = nullptr) {
    const float m_new = beta1 * m + (1.f - beta1) * g;
    const float v_new = beta2 * v + (1.f - beta2) * g * g;
    const float m_hat = m_new / (1.f - std::pow(beta1, static_cast<float>(t)));
    const float v_hat = v_new / (1.f - std::pow(beta2, static_cast<float>(t)));
    if (m_out) {
        *m_out = m_new;
    }
    if (v_out) {
        *v_out = v_new;
    }
    return theta - lr * m_hat / (std::sqrt(v_hat) + eps);
}

void test_adam_first_update() {
    // Step 1 bias correction: m_hat = g, v_hat = g^2.
    const float lr = 0.1f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float theta0 = 1.0f;
    const float g = 2.0f;

    learn::Variable w(learn::TensorF({1}, {theta0}), true);
    w.grad().at({0}) = g;

    learn::Adam opt({&w}, lr, beta1, beta2, eps);
    opt.step();

    const float expected = adam_expected(theta0, g, 0.f, 0.f, lr, beta1, beta2, eps, 1);
    // Explicitly: m_hat=g, v_hat=g*g on first step.
    const float expected_explicit = theta0 - lr * g / (std::sqrt(g * g) + eps);
    CHECK_NEAR(expected, expected_explicit, 1e-6f);
    CHECK_NEAR(w.data().at({0}), expected_explicit, 1e-5f);
}

void test_adam_multiple_updates() {
    const float lr = 0.05f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float theta0 = 3.0f;
    const float g1 = 1.5f;
    const float g2 = -0.5f;

    learn::Variable w(learn::TensorF({1}, {theta0}), true);
    learn::Adam opt({&w}, lr, beta1, beta2, eps);

    float m = 0.f;
    float v = 0.f;

    w.grad().at({0}) = g1;
    opt.step();
    float expected = adam_expected(theta0, g1, m, v, lr, beta1, beta2, eps, 1, &m, &v);
    CHECK_NEAR(w.data().at({0}), expected, 1e-5f);

    w.grad().at({0}) = g2;
    opt.step();
    expected = adam_expected(expected, g2, m, v, lr, beta1, beta2, eps, 2, &m, &v);
    CHECK_NEAR(w.data().at({0}), expected, 1e-5f);
}

void test_adam_multiple_parameters_independent_state() {
    const float lr = 0.1f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;

    learn::Variable w1(learn::TensorF({1}, {1.0f}), true);
    learn::Variable w2(learn::TensorF({1}, {2.0f}), true);
    w1.grad().at({0}) = 3.0f;
    w2.grad().at({0}) = -1.0f;

    learn::Adam opt({&w1, &w2}, lr, beta1, beta2, eps);
    opt.step();

    const float e1 = adam_expected(1.0f, 3.0f, 0.f, 0.f, lr, beta1, beta2, eps, 1);
    const float e2 = adam_expected(2.0f, -1.0f, 0.f, 0.f, lr, beta1, beta2, eps, 1);
    CHECK_NEAR(w1.data().at({0}), e1, 1e-5f);
    CHECK_NEAR(w2.data().at({0}), e2, 1e-5f);

    // Second step with swapped grads — each param's m/v must have tracked its own history.
    float m1 = 0.f;
    float v1 = 0.f;
    float m2 = 0.f;
    float v2 = 0.f;
    (void)adam_expected(1.0f, 3.0f, 0.f, 0.f, lr, beta1, beta2, eps, 1, &m1, &v1);
    (void)adam_expected(2.0f, -1.0f, 0.f, 0.f, lr, beta1, beta2, eps, 1, &m2, &v2);

    w1.grad().at({0}) = -1.0f;
    w2.grad().at({0}) = 3.0f;
    opt.step();

    const float e1b = adam_expected(e1, -1.0f, m1, v1, lr, beta1, beta2, eps, 2);
    const float e2b = adam_expected(e2, 3.0f, m2, v2, lr, beta1, beta2, eps, 2);
    CHECK_NEAR(w1.data().at({0}), e1b, 1e-5f);
    CHECK_NEAR(w2.data().at({0}), e2b, 1e-5f);
}

void test_adam_zero_grad() {
    const float lr = 0.1f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float theta0 = 1.0f;
    const float g1 = 2.0f;
    const float g2 = 0.5f;

    learn::Variable w(learn::TensorF({1}, {theta0}), true);
    learn::Adam opt({&w}, lr, beta1, beta2, eps);

    float m = 0.f;
    float v = 0.f;
    w.grad().at({0}) = g1;
    opt.step();
    float after_step1 = adam_expected(theta0, g1, m, v, lr, beta1, beta2, eps, 1, &m, &v);
    CHECK_NEAR(w.data().at({0}), after_step1, 1e-5f);

    w.grad().at({0}) = 99.f;
    opt.zero_grad();
    CHECK_NEAR(w.grad().at({0}), 0.f, 1e-7f);
    CHECK_NEAR(w.data().at({0}), after_step1, 1e-5f);  // parameters unchanged

    // Adam m/v state must still reflect step 1 (not reset by zero_grad).
    w.grad().at({0}) = g2;
    opt.step();
    const float after_step2 = adam_expected(after_step1, g2, m, v, lr, beta1, beta2, eps, 2);
    CHECK_NEAR(w.data().at({0}), after_step2, 1e-5f);
}

void test_adam_frozen_parameters() {
    learn::Variable frozen(learn::TensorF({2}, {4.f, 5.f}), /*requires_grad=*/false);
    learn::Variable trainable(learn::TensorF({1}, {1.f}), true);

    // Even if someone writes into grad storage, frozen params must not update.
    frozen.grad() = learn::TensorF({2}, {10.f, 20.f});
    trainable.grad().at({0}) = 3.f;

    const float frozen0 = frozen.data().at({0});
    const float frozen1 = frozen.data().at({1});

    learn::Adam opt({&frozen, &trainable}, 0.1f);
    opt.step();

    CHECK_NEAR(frozen.data().at({0}), frozen0, 1e-7f);
    CHECK_NEAR(frozen.data().at({1}), frozen1, 1e-7f);
    CHECK(trainable.data().at({0}) != 1.f);
}

void test_adam_different_tensor_shapes() {
    const float lr = 0.1f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;

    learn::Variable w1d(learn::TensorF({3}, {1.f, 2.f, 3.f}), true);
    learn::Variable w2d(learn::TensorF({2, 2}, {1.f, 1.f, 1.f, 1.f}), true);
    learn::Variable w3d(learn::TensorF({2, 1, 2}, {0.5f, -0.5f, 1.5f, -1.5f}), true);

    w1d.grad() = learn::TensorF({3}, {0.1f, 0.2f, 0.3f});
    w2d.grad() = learn::TensorF({2, 2}, {1.f, -1.f, 0.5f, -0.5f});
    w3d.grad() = learn::TensorF({2, 1, 2}, {2.f, 2.f, -2.f, -2.f});

    learn::Adam opt({&w1d, &w2d, &w3d}, lr, beta1, beta2, eps);
    opt.step();

    // Re-derive from known initial values (grads unchanged by step).
    const float o1[] = {1.f, 2.f, 3.f};
    const float g1[] = {0.1f, 0.2f, 0.3f};
    for (std::size_t i = 0; i < 3; ++i) {
        const float exp = adam_expected(o1[i], g1[i], 0.f, 0.f, lr, beta1, beta2, eps, 1);
        CHECK_NEAR(w1d.data().data()[i], exp, 1e-5f);
    }

    const float o2[] = {1.f, 1.f, 1.f, 1.f};
    const float g2[] = {1.f, -1.f, 0.5f, -0.5f};
    for (std::size_t i = 0; i < 4; ++i) {
        const float exp = adam_expected(o2[i], g2[i], 0.f, 0.f, lr, beta1, beta2, eps, 1);
        CHECK_NEAR(w2d.data().data()[i], exp, 1e-5f);
    }

    const float o3[] = {0.5f, -0.5f, 1.5f, -1.5f};
    const float g3[] = {2.f, 2.f, -2.f, -2.f};
    for (std::size_t i = 0; i < 4; ++i) {
        const float exp = adam_expected(o3[i], g3[i], 0.f, 0.f, lr, beta1, beta2, eps, 1);
        CHECK_NEAR(w3d.data().data()[i], exp, 1e-5f);
    }
}

void test_adam_invalid_configuration() {
    learn::Variable w(learn::TensorF({1}, {1.f}), true);

    CHECK(throws_exception([&] { learn::Adam({&w}, 0.f); }));
    CHECK(throws_exception([&] { learn::Adam({&w}, -0.1f); }));
    CHECK(throws_exception([&] { learn::Adam({&w}, 0.1f, 0.9f, 0.999f, 0.f); }));
    CHECK(throws_exception([&] { learn::Adam({&w}, 0.1f, 0.9f, 0.999f, -1e-8f); }));
    CHECK(throws_exception([&] { learn::Adam({&w}, 0.1f, -0.1f, 0.999f); }));
    CHECK(throws_exception([&] { learn::Adam({&w}, 0.1f, 1.0f, 0.999f); }));
    CHECK(throws_exception([&] { learn::Adam({&w}, 0.1f, 0.9f, -0.1f); }));
    CHECK(throws_exception([&] { learn::Adam({&w}, 0.1f, 0.9f, 1.0f); }));

    // Valid edge betas in [0, 1) should construct.
    CHECK(!throws_exception([&] { learn::Adam({&w}, 0.1f, 0.f, 0.f, 1e-8f); }));
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
    run_test("graph_lifetime", test_graph_lifetime);
    run_test("adam_first_update", test_adam_first_update);
    run_test("adam_multiple_updates", test_adam_multiple_updates);
    run_test("adam_multiple_parameters_independent_state", test_adam_multiple_parameters_independent_state);
    run_test("adam_zero_grad", test_adam_zero_grad);
    run_test("adam_frozen_parameters", test_adam_frozen_parameters);
    run_test("adam_different_tensor_shapes", test_adam_different_tensor_shapes);
    run_test("adam_invalid_configuration", test_adam_invalid_configuration);

    std::cout << tests_run << " checks, " << tests_failed << " failed\n";
    return tests_failed == 0 ? 0 : 1;
}
