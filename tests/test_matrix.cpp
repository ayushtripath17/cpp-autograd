#include "my_matrix.hpp"

#include <cmath>
#include <iostream>
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

#define CHECK_NEAR(a, b, eps) CHECK(std::abs((a) - (b)) < (eps))

bool matrices_equal(const learn::Matrix<double>& a, const learn::Matrix<double>& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            if (a(i, j) != b(i, j)) return false;
        }
    }
    return true;
}

void check_blocked_matches_matmul(const learn::Matrix<double>& a,
                                  const learn::Matrix<double>& b,
                                  std::size_t block_size) {
    const learn::Matrix<double> expected = a.matmul(b);
    const learn::Matrix<double> actual = a.blocked_matmul(b, block_size);
    CHECK(actual.rows() == expected.rows());
    CHECK(actual.cols() == expected.cols());
    CHECK(matrices_equal(actual, expected));
}

void test_blocked_matmul_known_result() {
    learn::Matrix<double> a(2, 3, {1, 2, 3, 4, 5, 6});
    learn::Matrix<double> b(3, 2, {7, 8, 9, 10, 11, 12});
    learn::Matrix<double> c = a.blocked_matmul(b, 2);

    CHECK(c.rows() == 2);
    CHECK(c.cols() == 2);
    CHECK(c(0, 0) == 58);
    CHECK(c(0, 1) == 64);
    CHECK(c(1, 0) == 139);
    CHECK(c(1, 1) == 154);
}

void test_blocked_matmul_identity() {
    learn::Matrix<double> a(2, 2, {1, 2, 3, 4});
    learn::Matrix<double> i(2, 2, {1, 0, 0, 1});
    learn::Matrix<double> c = a.blocked_matmul(i, 1);

    CHECK(c(0, 0) == 1);
    CHECK(c(0, 1) == 2);
    CHECK(c(1, 0) == 3);
    CHECK(c(1, 1) == 4);
}

void test_blocked_matmul_various_block_sizes() {
    learn::Matrix<double> a(5, 7, {
        1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28,
        29, 30, 31, 32, 33, 34, 35,
    });
    learn::Matrix<double> b(7, 4, {
        1, 0, 2, 1,
        0, 1, 1, 2,
        2, 1, 0, 0,
        1, 2, 3, 1,
        0, 0, 1, 2,
        3, 2, 1, 0,
        1, 1, 1, 1,
    });

    const std::vector<std::size_t> block_sizes = {1, 2, 3, 4, 5, 7, 8, 16, 32, 64};
    for (std::size_t block_size : block_sizes) {
        check_blocked_matches_matmul(a, b, block_size);
    }
}

void test_blocked_matmul_non_divisible_dimensions() {
    // Dimensions chosen so common block sizes leave partial edge tiles.
    learn::Matrix<double> a(3, 5, {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
    });
    learn::Matrix<double> b(5, 2, {
        1, 0,
        0, 1,
        2, 3,
        4, 5,
        6, 7,
    });

    check_blocked_matches_matmul(a, b, 2);
    check_blocked_matches_matmul(a, b, 3);
    check_blocked_matches_matmul(a, b, 4);
}

void test_blocked_matmul_rectangular_shapes() {
    learn::Matrix<double> a(1, 4, {1, 2, 3, 4});
    learn::Matrix<double> b(4, 3, {
        1, 0, 2,
        0, 1, 0,
        2, 2, 1,
        3, 1, 4,
    });
    check_blocked_matches_matmul(a, b, 2);

    learn::Matrix<double> c(4, 1, {1, 2, 3, 4});
    learn::Matrix<double> d(1, 3, {5, 6, 7});
    check_blocked_matches_matmul(c, d, 1);
}

void test_blocked_matmul_single_element() {
    learn::Matrix<double> a(1, 1, {6.0});
    learn::Matrix<double> b(1, 1, {7.0});
    check_blocked_matches_matmul(a, b, 1);
    check_blocked_matches_matmul(a, b, 8);
}

void test_blocked_matmul_zero_matrix() {
    learn::Matrix<double> a(2, 3, {0, 0, 0, 0, 0, 0});
    learn::Matrix<double> b(3, 2, {1, 2, 3, 4, 5, 6});
    check_blocked_matches_matmul(a, b, 2);
}

void test_blocked_matmul_larger_matrix() {
    learn::Matrix<double> a(8, 8);
    learn::Matrix<double> b(8, 8);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a.data()[i] = static_cast<double>(i) - 10.0;
        b.data()[i] = static_cast<double>(i % 5) + 0.5;
    }
    check_blocked_matches_matmul(a, b, 3);
    check_blocked_matches_matmul(a, b, 8);
}

void test_blocked_matmul_shape_mismatch() {
    learn::Matrix<double> a(2, 3);
    learn::Matrix<double> b(2, 2);
    bool threw = false;
    try {
        (void)a.blocked_matmul(b, 2);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

void test_default_construct() {
    learn::Matrix<double> m;
    CHECK(m.rows() == 0);
    CHECK(m.cols() == 0);
    CHECK(m.empty());
    CHECK(m.data().empty());
    CHECK(m.data().data() == nullptr);
}

void test_shape_constructor() {
    learn::Matrix<double> m(2, 3);
    CHECK(m.rows() == 2);
    CHECK(m.cols() == 3);
    CHECK(m.size() == 6);
    CHECK(m.data().size() == 6);
    CHECK(!m.empty());
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) {
            CHECK(m(i, j) == 0.0);
        }
    }
}

void test_fill_constructor() {
    learn::Matrix<double> m(2, 2, 3.5);
    CHECK(m.data().size() == 4);
    CHECK(m(0, 0) == 3.5);
    CHECK(m(1, 1) == 3.5);
}

void test_initializer_list() {
    learn::Matrix<double> m(2, 3, {1, 2, 3, 4, 5, 6});
    CHECK(m.data().size() == 6);
    CHECK(m(0, 0) == 1);
    CHECK(m(0, 2) == 3);
    CHECK(m(1, 0) == 4);
    CHECK(m(1, 2) == 6);
}

void test_element_access() {
    learn::Matrix<double> m(2, 2);
    m(0, 1) = 7.0;
    m(1, 0) = -2.0;
    CHECK(m(0, 1) == 7.0);
    CHECK(m(1, 0) == -2.0);
    CHECK(m.data()[1] == 7.0);  // row-major flat index
}

void test_const_access() {
    const learn::Matrix<double> m(2, 2, {1, 2, 3, 4});
    CHECK(m(1, 1) == 4);
    CHECK(m.data()[3] == 4);  // row-major: last element via Vector
    CHECK(m.data().data()[3] == 4);
}

void test_add() {
    learn::Matrix<double> a(2, 2, {1, 2, 3, 4});
    learn::Matrix<double> b(2, 2, {5, 6, 7, 8});
    learn::Matrix<double> c = a + b;

    CHECK(c(0, 0) == 6);
    CHECK(c(0, 1) == 8);
    CHECK(c(1, 0) == 10);
    CHECK(c(1, 1) == 12);
}

void test_subtract() {
    learn::Matrix<double> a(2, 2, {5, 6, 7, 8});
    learn::Matrix<double> b(2, 2, {1, 2, 3, 4});
    learn::Matrix<double> c = a - b;

    CHECK(c(0, 0) == 4);
    CHECK(c(1, 1) == 4);
}

void test_scalar_multiply() {
    learn::Matrix<double> m(2, 2, {1, 2, 3, 4});
    learn::Matrix<double> c = m * 2.0;

    CHECK(c(0, 0) == 2);
    CHECK(c(1, 1) == 8);
}

void test_transpose() {
    learn::Matrix<double> m(2, 3, {1, 2, 3, 4, 5, 6});
    learn::Matrix<double> t = m.transpose();

    CHECK(t.rows() == 3);
    CHECK(t.cols() == 2);
    CHECK(t.data().size() == 6);
    CHECK(t(0, 0) == 1);
    CHECK(t(0, 1) == 4);
    CHECK(t(2, 1) == 6);
}

void test_matmul() {
    learn::Matrix<double> a(2, 3, {1, 2, 3, 4, 5, 6});
    learn::Matrix<double> b(3, 2, {7, 8, 9, 10, 11, 12});
    learn::Matrix<double> c = a.matmul(b);

    CHECK(c.rows() == 2);
    CHECK(c.cols() == 2);
    CHECK(c.data().size() == 4);
    CHECK(c(0, 0) == 58);
    CHECK(c(0, 1) == 64);
    CHECK(c(1, 0) == 139);
    CHECK(c(1, 1) == 154);
}

void test_matmul_identity() {
    learn::Matrix<double> a(2, 2, {1, 2, 3, 4});
    learn::Matrix<double> i(2, 2, {1, 0, 0, 1});
    learn::Matrix<double> c = a.matmul(i);

    CHECK(c(0, 0) == 1);
    CHECK(c(0, 1) == 2);
    CHECK(c(1, 0) == 3);
    CHECK(c(1, 1) == 4);
}

void test_copy_constructor_deep_copy() {
    learn::Matrix<double> a(2, 2, {1, 2, 3, 4});
    learn::Matrix<double> b = a;

    CHECK(b.rows() == 2);
    CHECK(b.cols() == 2);
    CHECK(b.data().size() == a.data().size());
    CHECK(b(0, 0) == 1);
    CHECK(b(1, 1) == 4);
    CHECK(b.data().data() != a.data().data());  // distinct Vector storage

    a(0, 0) = 99;
    CHECK(b(0, 0) == 1);
}

void test_copy_assignment() {
    learn::Matrix<double> a(2, 2, {1, 2, 3, 4});
    learn::Matrix<double> b(1, 1);
    b = a;

    CHECK(b.rows() == 2);
    CHECK(b.cols() == 2);
    CHECK(b(1, 1) == 4);
    CHECK(b.data().data() != a.data().data());
}

void test_copy_assignment_self() {
    learn::Matrix<double> a(2, 2, {1, 2, 3, 4});
    learn::Matrix<double>& ref = a;
    a = ref;
    CHECK(a(1, 1) == 4);
    CHECK(a.data().size() == 4);
}

void test_move_constructor() {
    learn::Matrix<double> a(2, 2, {1, 2, 3, 4});
    const double* old_buf = a.data().data();

    learn::Matrix<double> b = std::move(a);

    CHECK(b.rows() == 2);
    CHECK(b.cols() == 2);
    CHECK(b(0, 0) == 1);
    CHECK(b(1, 1) == 4);
    CHECK(b.data().data() == old_buf);  // stole Vector's buffer

    // moved-from a: valid but unspecified — must remain usable
    a = learn::Matrix<double>(1, 1, {99.0});
    CHECK(a.rows() == 1);
    CHECK(a.cols() == 1);
    CHECK(a(0, 0) == 99.0);
}

void test_move_assignment() {
    learn::Matrix<double> a(2, 2, {1, 2, 3, 4});
    learn::Matrix<double> b(1, 1, {9});
    const double* old_buf = a.data().data();

    b = std::move(a);

    CHECK(b.rows() == 2);
    CHECK(b.cols() == 2);
    CHECK(b(0, 1) == 2);
    CHECK(b.data().data() == old_buf);

    // moved-from a: valid but unspecified — must remain usable
    a = learn::Matrix<double>(2, 1, {1.0, 2.0});
    CHECK(a.rows() == 2);
    CHECK(a.cols() == 1);
    CHECK(a(1, 0) == 2.0);
}

void test_add_shape_mismatch() {
    learn::Matrix<double> a(2, 2);
    learn::Matrix<double> b(2, 3);
    bool threw = false;
    try {
        (void)(a + b);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

void test_matmul_shape_mismatch() {
    learn::Matrix<double> a(2, 3);
    learn::Matrix<double> b(2, 2);
    bool threw = false;
    try {
        (void)a.matmul(b);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

void test_initializer_list_size_mismatch() {
    bool threw = false;
    try {
        learn::Matrix<double> m(2, 2, {1, 2, 3});
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

}  // namespace

int main() {
    std::cout << "Running Matrix exercise tests...\n\n";

    run_test("test_default_construct", test_default_construct);
    run_test("test_shape_constructor", test_shape_constructor);
    run_test("test_fill_constructor", test_fill_constructor);
    run_test("test_initializer_list", test_initializer_list);
    run_test("test_element_access", test_element_access);
    run_test("test_const_access", test_const_access);
    run_test("test_add", test_add);
    run_test("test_subtract", test_subtract);
    run_test("test_scalar_multiply", test_scalar_multiply);
    run_test("test_transpose", test_transpose);
    run_test("test_matmul", test_matmul);
    run_test("test_matmul_identity", test_matmul_identity);
    run_test("test_copy_constructor_deep_copy", test_copy_constructor_deep_copy);
    run_test("test_copy_assignment", test_copy_assignment);
    run_test("test_copy_assignment_self", test_copy_assignment_self);
    run_test("test_move_constructor", test_move_constructor);
    run_test("test_move_assignment", test_move_assignment);
    run_test("test_add_shape_mismatch", test_add_shape_mismatch);
    run_test("test_matmul_shape_mismatch", test_matmul_shape_mismatch);
    run_test("test_initializer_list_size_mismatch", test_initializer_list_size_mismatch);
    run_test("test_blocked_matmul_known_result", test_blocked_matmul_known_result);
    run_test("test_blocked_matmul_identity", test_blocked_matmul_identity);
    run_test("test_blocked_matmul_various_block_sizes", test_blocked_matmul_various_block_sizes);
    run_test("test_blocked_matmul_non_divisible_dimensions", test_blocked_matmul_non_divisible_dimensions);
    run_test("test_blocked_matmul_rectangular_shapes", test_blocked_matmul_rectangular_shapes);
    run_test("test_blocked_matmul_single_element", test_blocked_matmul_single_element);
    run_test("test_blocked_matmul_zero_matrix", test_blocked_matmul_zero_matrix);
    run_test("test_blocked_matmul_larger_matrix", test_blocked_matmul_larger_matrix);
    run_test("test_blocked_matmul_shape_mismatch", test_blocked_matmul_shape_mismatch);

    std::cout << "\n" << tests_run << " checks, " << tests_failed << " failed.\n";

    if (tests_failed == 0) {
        std::cout << "All tests passed!\n";
        return 0;
    }

    std::cout << "Some tests failed — keep implementing my_matrix.hpp.\n";
    return 1;
}
