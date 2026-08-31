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

void test_matmul_into_matches_matmul() {
    learn::Matrix<double> a(2, 3, {1, 2, 3, 4, 5, 6});
    learn::Matrix<double> b(3, 2, {7, 8, 9, 10, 11, 12});

    learn::Matrix<double> expected = a.matmul(b);
    learn::Matrix<double> actual(2, 2, 999.0);
    a.matmul_into(actual, b);

    CHECK(matrices_equal(actual, expected));
}

void test_matmul_into_shape_mismatch() {
    learn::Matrix<double> a(2, 3);
    learn::Matrix<double> b(3, 2);
    learn::Matrix<double> out(2, 3);
    bool threw = false;
    try {
        a.matmul_into(out, b);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
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

void check_multithreaded_matches_matmul(const learn::Matrix<double>& a,
                                        const learn::Matrix<double>& b,
                                        std::size_t num_threads) {
    const learn::Matrix<double> expected = a.matmul(b);
    const learn::Matrix<double> actual = a.multithreaded_matmul(b, num_threads);
    CHECK(actual.rows() == expected.rows());
    CHECK(actual.cols() == expected.cols());
    CHECK(matrices_equal(actual, expected));
}

void test_multithreaded_matmul_known_result() {
    learn::Matrix<double> a(2, 3, {1, 2, 3, 4, 5, 6});
    learn::Matrix<double> b(3, 2, {7, 8, 9, 10, 11, 12});
    learn::Matrix<double> c = a.multithreaded_matmul(b, 2);

    CHECK(c.rows() == 2);
    CHECK(c.cols() == 2);
    CHECK(c(0, 0) == 58);
    CHECK(c(0, 1) == 64);
    CHECK(c(1, 0) == 139);
    CHECK(c(1, 1) == 154);
}

void test_multithreaded_matmul_identity() {
    learn::Matrix<double> a(2, 2, {1, 2, 3, 4});
    learn::Matrix<double> i(2, 2, {1, 0, 0, 1});
    learn::Matrix<double> c = a.multithreaded_matmul(i, 1);

    CHECK(c(0, 0) == 1);
    CHECK(c(0, 1) == 2);
    CHECK(c(1, 0) == 3);
    CHECK(c(1, 1) == 4);
}

void test_multithreaded_matmul_various_thread_counts() {
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

    const std::vector<std::size_t> thread_counts = {1, 2, 3, 4, 5, 8, 16};
    for (std::size_t num_threads : thread_counts) {
        check_multithreaded_matches_matmul(a, b, num_threads);
    }
}

void test_multithreaded_matmul_rows_not_divisible_by_threads() {
    // 5 rows with 2/3/4 threads leaves uneven row partitions.
    learn::Matrix<double> a(5, 3, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15,
    });
    learn::Matrix<double> b(3, 2, {
        1, 0,
        0, 1,
        2, 3,
    });

    check_multithreaded_matches_matmul(a, b, 2);
    check_multithreaded_matches_matmul(a, b, 3);
    check_multithreaded_matches_matmul(a, b, 4);
}

void test_multithreaded_matmul_more_threads_than_rows() {
    learn::Matrix<double> a(2, 3, {1, 2, 3, 4, 5, 6});
    learn::Matrix<double> b(3, 2, {7, 8, 9, 10, 11, 12});
    check_multithreaded_matches_matmul(a, b, 8);
}

void test_multithreaded_matmul_rectangular_shapes() {
    learn::Matrix<double> a(1, 4, {1, 2, 3, 4});
    learn::Matrix<double> b(4, 3, {
        1, 0, 2,
        0, 1, 0,
        2, 2, 1,
        3, 1, 4,
    });
    check_multithreaded_matches_matmul(a, b, 2);

    learn::Matrix<double> c(4, 1, {1, 2, 3, 4});
    learn::Matrix<double> d(1, 3, {5, 6, 7});
    check_multithreaded_matches_matmul(c, d, 4);
}

void test_multithreaded_matmul_single_element() {
    learn::Matrix<double> a(1, 1, {6.0});
    learn::Matrix<double> b(1, 1, {7.0});
    check_multithreaded_matches_matmul(a, b, 1);
    check_multithreaded_matches_matmul(a, b, 4);
}

void test_multithreaded_matmul_zero_matrix() {
    learn::Matrix<double> a(2, 3, {0, 0, 0, 0, 0, 0});
    learn::Matrix<double> b(3, 2, {1, 2, 3, 4, 5, 6});
    check_multithreaded_matches_matmul(a, b, 2);
}

void test_multithreaded_matmul_larger_matrix() {
    learn::Matrix<double> a(16, 16);
    learn::Matrix<double> b(16, 16);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a.data()[i] = static_cast<double>(i) - 10.0;
        b.data()[i] = static_cast<double>(i % 5) + 0.5;
    }
    check_multithreaded_matches_matmul(a, b, 1);
    check_multithreaded_matches_matmul(a, b, 2);
    check_multithreaded_matches_matmul(a, b, 4);
    check_multithreaded_matches_matmul(a, b, 8);
}

void test_multithreaded_matmul_shape_mismatch() {
    learn::Matrix<double> a(2, 3);
    learn::Matrix<double> b(2, 2);
    bool threw = false;
    try {
        (void)a.multithreaded_matmul(b, 2);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

void check_register_optimized_matches_matmul(const learn::Matrix<double>& a,
                                             const learn::Matrix<double>& b) {
    const learn::Matrix<double> expected = a.matmul(b);
    const learn::Matrix<double> actual = a.register_optimized_matmul(b);
    CHECK(actual.rows() == expected.rows());
    CHECK(actual.cols() == expected.cols());
    CHECK(matrices_equal(actual, expected));
}

void test_register_optimized_matmul_known_result() {
    learn::Matrix<double> a(2, 3, {1, 2, 3, 4, 5, 6});
    learn::Matrix<double> b(3, 2, {7, 8, 9, 10, 11, 12});
    learn::Matrix<double> c = a.register_optimized_matmul(b);

    CHECK(c.rows() == 2);
    CHECK(c.cols() == 2);
    CHECK(c(0, 0) == 58);
    CHECK(c(0, 1) == 64);
    CHECK(c(1, 0) == 139);
    CHECK(c(1, 1) == 154);
}

void test_register_optimized_matmul_identity() {
    learn::Matrix<double> a(2, 2, {1, 2, 3, 4});
    learn::Matrix<double> i(2, 2, {1, 0, 0, 1});
    learn::Matrix<double> c = a.register_optimized_matmul(i);

    CHECK(c(0, 0) == 1);
    CHECK(c(0, 1) == 2);
    CHECK(c(1, 0) == 3);
    CHECK(c(1, 1) == 4);
}

void test_register_optimized_matmul_odd_rows() {
    // Tile height is 2; odd row count leaves a partial bottom tile.
    learn::Matrix<double> a(3, 4, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
    });
    learn::Matrix<double> b(4, 4, {
        1, 0, 0, 1,
        0, 1, 1, 0,
        2, 0, 1, 0,
        0, 2, 0, 1,
    });
    check_register_optimized_matches_matmul(a, b);
}

void test_register_optimized_matmul_cols_not_multiple_of_four() {
    // Tile width is 4; leftover columns on the right edge.
    learn::Matrix<double> a(4, 3, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
    });
    learn::Matrix<double> b(3, 5, {
        1, 0, 2, 1, 0,
        0, 1, 1, 0, 2,
        2, 1, 0, 1, 1,
    });
    check_register_optimized_matches_matmul(a, b);
}

void test_register_optimized_matmul_rectangular_shapes() {
    learn::Matrix<double> a(1, 4, {1, 2, 3, 4});
    learn::Matrix<double> b(4, 3, {
        1, 0, 2,
        0, 1, 0,
        2, 2, 1,
        3, 1, 4,
    });
    check_register_optimized_matches_matmul(a, b);

    learn::Matrix<double> c(4, 1, {1, 2, 3, 4});
    learn::Matrix<double> d(1, 3, {5, 6, 7});
    check_register_optimized_matches_matmul(c, d);
}

void test_register_optimized_matmul_single_element() {
    learn::Matrix<double> a(1, 1, {6.0});
    learn::Matrix<double> b(1, 1, {7.0});
    check_register_optimized_matches_matmul(a, b);
}

void test_register_optimized_matmul_zero_matrix() {
    learn::Matrix<double> a(2, 3, {0, 0, 0, 0, 0, 0});
    learn::Matrix<double> b(3, 2, {1, 2, 3, 4, 5, 6});
    check_register_optimized_matches_matmul(a, b);
}

void test_register_optimized_matmul_larger_matrix() {
    // Mix of full 2x4 tiles and leftover edges (7x5 @ 5x6).
    learn::Matrix<double> a(7, 5);
    learn::Matrix<double> b(5, 6);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a.data()[i] = static_cast<double>(i) - 3.0;
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        b.data()[i] = static_cast<double>(i % 7) + 0.25;
    }
    check_register_optimized_matches_matmul(a, b);
}

void test_register_optimized_matmul_shape_mismatch() {
    learn::Matrix<double> a(2, 3);
    learn::Matrix<double> b(2, 2);
    bool threw = false;
    try {
        (void)a.register_optimized_matmul(b);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

void check_register_packing_matches_matmul(const learn::Matrix<double>& a,
                                           const learn::Matrix<double>& b) {
    const learn::Matrix<double> expected = a.matmul(b);
    const learn::Matrix<double> actual = a.register_optimized_matmul_packing(b);
    CHECK(actual.rows() == expected.rows());
    CHECK(actual.cols() == expected.cols());
    CHECK(matrices_equal(actual, expected));
}

void test_register_packing_matmul_known_result() {
    learn::Matrix<double> a(2, 3, {1, 2, 3, 4, 5, 6});
    learn::Matrix<double> b(3, 2, {7, 8, 9, 10, 11, 12});
    learn::Matrix<double> c = a.register_optimized_matmul_packing(b);

    CHECK(c.rows() == 2);
    CHECK(c.cols() == 2);
    CHECK(c(0, 0) == 58);
    CHECK(c(0, 1) == 64);
    CHECK(c(1, 0) == 139);
    CHECK(c(1, 1) == 154);
}

void test_register_packing_matmul_identity() {
    learn::Matrix<double> a(2, 2, {1, 2, 3, 4});
    learn::Matrix<double> i(2, 2, {1, 0, 0, 1});
    learn::Matrix<double> c = a.register_optimized_matmul_packing(i);

    CHECK(c(0, 0) == 1);
    CHECK(c(0, 1) == 2);
    CHECK(c(1, 0) == 3);
    CHECK(c(1, 1) == 4);
}

void test_register_packing_matmul_single_element() {
    learn::Matrix<double> a(1, 1, {6.0});
    learn::Matrix<double> b(1, 1, {7.0});
    check_register_packing_matches_matmul(a, b);
}

void test_register_packing_matmul_zero_matrix() {
    learn::Matrix<double> a(2, 3, {0, 0, 0, 0, 0, 0});
    learn::Matrix<double> b(3, 2, {1, 2, 3, 4, 5, 6});
    check_register_packing_matches_matmul(a, b);
}

void test_register_packing_matmul_odd_rows() {
    // Tile height is 2; odd row count leaves a partial bottom tile.
    learn::Matrix<double> a(3, 4, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
    });
    learn::Matrix<double> b(4, 4, {
        1, 0, 0, 1,
        0, 1, 1, 0,
        2, 0, 1, 0,
        0, 2, 0, 1,
    });
    check_register_packing_matches_matmul(a, b);
}

void test_register_packing_matmul_cols_not_multiple_of_four() {
    // Packing pads B panels to width 4; leftover columns must still match matmul.
    learn::Matrix<double> a(4, 3, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
    });
    learn::Matrix<double> b(3, 5, {
        1, 0, 2, 1, 0,
        0, 1, 1, 0, 2,
        2, 1, 0, 1, 1,
    });
    check_register_packing_matches_matmul(a, b);
}

void test_register_packing_matmul_cols_exactly_four() {
    // Full panels only — no leftover padding path.
    learn::Matrix<double> a(2, 3, {1, 2, 3, 4, 5, 6});
    learn::Matrix<double> b(3, 4, {
        1, 0, 2, 1,
        0, 1, 1, 0,
        2, 1, 0, 3,
    });
    check_register_packing_matches_matmul(a, b);
}

void test_register_packing_matmul_rectangular_shapes() {
    learn::Matrix<double> a(1, 4, {1, 2, 3, 4});
    learn::Matrix<double> b(4, 3, {
        1, 0, 2,
        0, 1, 0,
        2, 2, 1,
        3, 1, 4,
    });
    check_register_packing_matches_matmul(a, b);

    learn::Matrix<double> c(4, 1, {1, 2, 3, 4});
    learn::Matrix<double> d(1, 3, {5, 6, 7});
    check_register_packing_matches_matmul(c, d);
}

void test_register_packing_matmul_larger_matrix() {
    // Mix of full panels and leftovers (7x5 @ 5x6).
    learn::Matrix<double> a(7, 5);
    learn::Matrix<double> b(5, 6);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a.data()[i] = static_cast<double>(i) - 3.0;
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        b.data()[i] = static_cast<double>(i % 7) + 0.25;
    }
    check_register_packing_matches_matmul(a, b);
}

void test_register_packing_matmul_matches_register_optimized() {
    learn::Matrix<double> a(5, 7);
    learn::Matrix<double> b(7, 5);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a.data()[i] = static_cast<double>(i) * 0.5 - 1.0;
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        b.data()[i] = static_cast<double>((i * 3) % 11) - 2.0;
    }
    const learn::Matrix<double> tiled = a.register_optimized_matmul(b);
    const learn::Matrix<double> packed = a.register_optimized_matmul_packing(b);
    CHECK(matrices_equal(tiled, packed));
}

void test_register_packing_matmul_shape_mismatch() {
    learn::Matrix<double> a(2, 3);
    learn::Matrix<double> b(2, 2);
    bool threw = false;
    try {
        (void)a.register_optimized_matmul_packing(b);
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
    run_test("test_matmul_into_matches_matmul", test_matmul_into_matches_matmul);
    run_test("test_matmul_into_shape_mismatch", test_matmul_into_shape_mismatch);
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
    run_test("test_multithreaded_matmul_known_result", test_multithreaded_matmul_known_result);
    run_test("test_multithreaded_matmul_identity", test_multithreaded_matmul_identity);
    run_test("test_multithreaded_matmul_various_thread_counts", test_multithreaded_matmul_various_thread_counts);
    run_test("test_multithreaded_matmul_rows_not_divisible_by_threads", test_multithreaded_matmul_rows_not_divisible_by_threads);
    run_test("test_multithreaded_matmul_more_threads_than_rows", test_multithreaded_matmul_more_threads_than_rows);
    run_test("test_multithreaded_matmul_rectangular_shapes", test_multithreaded_matmul_rectangular_shapes);
    run_test("test_multithreaded_matmul_single_element", test_multithreaded_matmul_single_element);
    run_test("test_multithreaded_matmul_zero_matrix", test_multithreaded_matmul_zero_matrix);
    run_test("test_multithreaded_matmul_larger_matrix", test_multithreaded_matmul_larger_matrix);
    run_test("test_multithreaded_matmul_shape_mismatch", test_multithreaded_matmul_shape_mismatch);
    run_test("test_register_optimized_matmul_known_result", test_register_optimized_matmul_known_result);
    run_test("test_register_optimized_matmul_identity", test_register_optimized_matmul_identity);
    run_test("test_register_optimized_matmul_odd_rows", test_register_optimized_matmul_odd_rows);
    run_test("test_register_optimized_matmul_cols_not_multiple_of_four", test_register_optimized_matmul_cols_not_multiple_of_four);
    run_test("test_register_optimized_matmul_rectangular_shapes", test_register_optimized_matmul_rectangular_shapes);
    run_test("test_register_optimized_matmul_single_element", test_register_optimized_matmul_single_element);
    run_test("test_register_optimized_matmul_zero_matrix", test_register_optimized_matmul_zero_matrix);
    run_test("test_register_optimized_matmul_larger_matrix", test_register_optimized_matmul_larger_matrix);
    run_test("test_register_optimized_matmul_shape_mismatch", test_register_optimized_matmul_shape_mismatch);
    run_test("test_register_packing_matmul_known_result", test_register_packing_matmul_known_result);
    run_test("test_register_packing_matmul_identity", test_register_packing_matmul_identity);
    run_test("test_register_packing_matmul_single_element", test_register_packing_matmul_single_element);
    run_test("test_register_packing_matmul_zero_matrix", test_register_packing_matmul_zero_matrix);
    run_test("test_register_packing_matmul_odd_rows", test_register_packing_matmul_odd_rows);
    run_test("test_register_packing_matmul_cols_not_multiple_of_four", test_register_packing_matmul_cols_not_multiple_of_four);
    run_test("test_register_packing_matmul_cols_exactly_four", test_register_packing_matmul_cols_exactly_four);
    run_test("test_register_packing_matmul_rectangular_shapes", test_register_packing_matmul_rectangular_shapes);
    run_test("test_register_packing_matmul_larger_matrix", test_register_packing_matmul_larger_matrix);
    run_test("test_register_packing_matmul_matches_register_optimized", test_register_packing_matmul_matches_register_optimized);
    run_test("test_register_packing_matmul_shape_mismatch", test_register_packing_matmul_shape_mismatch);

    std::cout << "\n" << tests_run << " checks, " << tests_failed << " failed.\n";

    if (tests_failed == 0) {
        std::cout << "All tests passed!\n";
        return 0;
    }

    std::cout << "Some tests failed — keep implementing my_matrix.hpp.\n";
    return 1;
}
