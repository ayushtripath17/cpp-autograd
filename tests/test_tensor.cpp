#include "my_tensor.hpp"

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

void test_default_construct() {
    learn::Tensor<double> t;
    CHECK(t.empty());
    CHECK(t.ndim() == 0);
    CHECK(t.size() == 0);
}

void test_shape_constructor_zeros() {
    learn::Tensor<double> t({2, 3});
    CHECK(t.ndim() == 2);
    CHECK(t.size() == 6);
    CHECK(t.shape()[0] == 2);
    CHECK(t.shape()[1] == 3);
    CHECK(t.at({0, 0}) == 0.0);
    CHECK(t.at({1, 2}) == 0.0);
}

void test_fill_constructor() {
    learn::Tensor<double> t({2, 2}, 3.5);
    CHECK(t.at({0, 0}) == 3.5);
    CHECK(t.at({1, 1}) == 3.5);
}

void test_values_constructor() {
    learn::Tensor<double> t({2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK(t.at({0, 0}) == 1);
    CHECK(t.at({0, 2}) == 3);
    CHECK(t.at({1, 0}) == 4);
    CHECK(t.at({1, 2}) == 6);
}

void test_strides_row_major() {
    learn::Tensor<double> t({2, 3, 4});
    // shape [2,3,4] -> strides [12, 4, 1]
    CHECK(t.strides()[0] == 12);
    CHECK(t.strides()[1] == 4);
    CHECK(t.strides()[2] == 1);
}

void test_element_access_write() {
    learn::Tensor<double> t({2, 2});
    t.at({0, 1}) = 7.0;
    t.at({1, 0}) = -2.0;
    CHECK(t.at({0, 1}) == 7.0);
    CHECK(t.at({1, 0}) == -2.0);
}

void test_reshape() {
    learn::Tensor<double> t({2, 3}, {1, 2, 3, 4, 5, 6});
    learn::Tensor<double> r = t.reshape({3, 2});

    CHECK(r.ndim() == 2);
    CHECK(r.shape()[0] == 3);
    CHECK(r.shape()[1] == 2);
    CHECK(r.size() == 6);
    CHECK(r.at({0, 0}) == 1);
    CHECK(r.at({2, 1}) == 6);
}

void test_transpose_2d() {
    learn::Tensor<double> t({2, 3}, {1, 2, 3, 4, 5, 6});
    learn::Tensor<double> tt = t.transpose(0, 1);

    CHECK(tt.shape()[0] == 3);
    CHECK(tt.shape()[1] == 2);
    CHECK(tt.at({0, 0}) == 1);
    CHECK(tt.at({0, 1}) == 4);
    CHECK(tt.at({2, 1}) == 6);
}

void test_transpose_3d_swap_axes() {
    learn::Tensor<double> t({2, 3, 1});
    t.at({0, 0, 0}) = 1;
    t.at({1, 2, 0}) = 6;

    learn::Tensor<double> tt = t.transpose(0, 2);
    CHECK(tt.shape()[0] == 1);
    CHECK(tt.shape()[1] == 3);
    CHECK(tt.shape()[2] == 2);
    CHECK(tt.at({0, 0, 0}) == 1);
    CHECK(tt.at({0, 2, 1}) == 6);
}

void test_view_shares_storage() {
    learn::Tensor<double> t({2, 2}, {1, 2, 3, 4});
    learn::TensorView<double> v = t.view({4});

    CHECK(v.size() == 4);
    CHECK(v.at({0}) == 1);
    v.at({2}) = 99.0;
    CHECK(t.at({1, 0}) == 99.0);
}

void test_add_same_shape() {
    learn::Tensor<double> a({2, 2}, {1, 2, 3, 4});
    learn::Tensor<double> b({2, 2}, {5, 6, 7, 8});
    learn::Tensor<double> c = a + b;

    CHECK(c.at({0, 0}) == 6);
    CHECK(c.at({1, 1}) == 12);
}

void test_matmul_2d() {
    learn::Tensor<double> a({2, 3}, {1, 2, 3, 4, 5, 6});
    learn::Tensor<double> b({3, 2}, {7, 8, 9, 10, 11, 12});
    learn::Tensor<double> c = a.matmul(b);

    CHECK(c.shape()[0] == 2);
    CHECK(c.shape()[1] == 2);
    CHECK(c.at({0, 0}) == 58);
    CHECK(c.at({1, 1}) == 154);
}

void test_matmul_batched() {
    // One batch: (1, 2, 3) @ (1, 3, 2) -> (1, 2, 2)
    learn::Tensor<double> a({1, 2, 3}, {1, 2, 3, 4, 5, 6});
    learn::Tensor<double> b({1, 3, 2}, {7, 8, 9, 10, 11, 12});
    learn::Tensor<double> c = a.matmul(b);

    CHECK(c.ndim() == 3);
    CHECK(c.shape()[0] == 1);
    CHECK(c.shape()[1] == 2);
    CHECK(c.shape()[2] == 2);
    CHECK(c.at({0, 0, 0}) == 58);
    CHECK(c.at({0, 1, 1}) == 154);
}

void test_matmul_two_batches() {
    // Batch dim 2: each slice is (2x3) @ (3x2)
    learn::Tensor<double> a({2, 2, 3});
    learn::Tensor<double> b({2, 3, 2});
    a.at({0, 0, 0}) = 1;
    a.at({0, 1, 2}) = 6;
    a.at({1, 0, 0}) = 10;
    a.at({1, 1, 2}) = 15;

    b.at({0, 0, 0}) = 1;
    b.at({0, 2, 1}) = 1;
    b.at({1, 0, 0}) = 2;
    b.at({1, 2, 1}) = 2;

    learn::Tensor<double> c = a.matmul(b);
    CHECK(c.shape()[0] == 2);
    CHECK(c.shape()[1] == 2);
    CHECK(c.shape()[2] == 2);
    CHECK(c.at({0, 0, 0}) == 1);
    CHECK(c.at({1, 1, 1}) == 30);
}

void test_reshape_invalid() {
    learn::Tensor<double> t({2, 3});
    bool threw = false;
    try {
        (void)t.reshape({2, 2});
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

void test_add_shape_mismatch() {
    learn::Tensor<double> a({2, 2});
    learn::Tensor<double> b({2, 3});
    bool threw = false;
    try {
        (void)(a + b);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

void test_matmul_shape_mismatch() {
    learn::Tensor<double> a({2, 3});
    learn::Tensor<double> b({2, 2});
    bool threw = false;
    try {
        (void)a.matmul(b);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

}  // namespace

int main() {
    std::cout << "Running Tensor exercise tests...\n\n";

    run_test("test_default_construct", test_default_construct);
    run_test("test_shape_constructor_zeros", test_shape_constructor_zeros);
    run_test("test_fill_constructor", test_fill_constructor);
    run_test("test_values_constructor", test_values_constructor);
    run_test("test_strides_row_major", test_strides_row_major);
    run_test("test_element_access_write", test_element_access_write);
    run_test("test_reshape", test_reshape);
    run_test("test_transpose_2d", test_transpose_2d);
    run_test("test_transpose_3d_swap_axes", test_transpose_3d_swap_axes);
    run_test("test_view_shares_storage", test_view_shares_storage);
    run_test("test_add_same_shape", test_add_same_shape);
    run_test("test_matmul_2d", test_matmul_2d);
    run_test("test_matmul_batched", test_matmul_batched);
    run_test("test_matmul_two_batches", test_matmul_two_batches);
    run_test("test_reshape_invalid", test_reshape_invalid);
    run_test("test_add_shape_mismatch", test_add_shape_mismatch);
    run_test("test_matmul_shape_mismatch", test_matmul_shape_mismatch);

    std::cout << "\n" << tests_run << " checks, " << tests_failed << " failed.\n";

    if (tests_failed == 0) {
        std::cout << "All tests passed!\n";
        return 0;
    }

    std::cout << "Some tests failed — keep implementing my_tensor.hpp.\n";
    return 1;
}
