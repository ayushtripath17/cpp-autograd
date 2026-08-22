#include "my_vector.hpp"

#include <cassert>
#include <iostream>
#include <string>
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
    learn::Vector<int> v;
    CHECK(v.size() == 0);
    CHECK(v.capacity() == 0);
    CHECK(v.empty());
    CHECK(v.data() == nullptr);
}

void test_count_constructor() {
    learn::Vector<int> v(5);
    CHECK(v.size() == 5);
    CHECK(v.capacity() >= 5);
    for (std::size_t i = 0; i < v.size(); ++i) {
        CHECK(v[i] == 0);
    }
}

void test_count_value_constructor() {
    learn::Vector<int> v(4, 7);
    CHECK(v.size() == 4);
    for (std::size_t i = 0; i < v.size(); ++i) {
        CHECK(v[i] == 7);
    }
}

void test_initializer_list() {
    learn::Vector<int> v = {1, 2, 3, 4, 5};
    CHECK(v.size() == 5);
    CHECK(v[0] == 1);
    CHECK(v[4] == 5);
}

void test_push_back_and_reallocation() {
    learn::Vector<int> v;
    for (int i = 0; i < 100; ++i) {
        v.push_back(i);
    }
    CHECK(v.size() == 100);
    CHECK(v.capacity() >= 100);
    for (int i = 0; i < 100; ++i) {
        CHECK(v[i] == i);
    }
}

void test_copy_constructor_deep_copy() {
    learn::Vector<int> a = {1, 2, 3};
    learn::Vector<int> b = a;

    CHECK(b.size() == 3);
    CHECK(b[0] == 1 && b[2] == 3);
    CHECK(b.data() != a.data());  // distinct storage

    a[0] = 99;
    CHECK(b[0] == 1);  // b unchanged
}

void test_copy_assignment() {
    learn::Vector<int> a = {1, 2, 3};
    learn::Vector<int> b;
    b = a;

    CHECK(b.size() == 3);
    CHECK(b[1] == 2);
    CHECK(b.data() != a.data());

    a[1] = 42;
    CHECK(b[1] == 2);
}

void test_copy_assignment_self() {
    learn::Vector<int> a = {1, 2, 3};
    learn::Vector<int>& ref = a;
    a = ref;
    CHECK(a.size() == 3);
    CHECK(a[2] == 3);
}

void test_move_constructor() {
    learn::Vector<int> a = {1, 2, 3};
    const int* old_data = a.data();

    learn::Vector<int> b = std::move(a);

    CHECK(b.size() == 3);
    CHECK(b[2] == 3);
    CHECK(b.data() == old_data);  // stole buffer
    CHECK(a.size() == 0);
    CHECK(a.data() == nullptr);
}

void test_move_assignment() {
    learn::Vector<int> a = {1, 2, 3};
    learn::Vector<int> b = {9, 9};
    const int* old_data = a.data();

    b = std::move(a);

    CHECK(b.size() == 3);
    CHECK(b[0] == 1);
    CHECK(b.data() == old_data);
    CHECK(a.size() == 0);
    CHECK(a.data() == nullptr);
}

void test_reserve() {
    learn::Vector<int> v;
    v.reserve(64);
    CHECK(v.capacity() >= 64);
    CHECK(v.size() == 0);
    CHECK(v.empty());
}

void test_resize() {
    learn::Vector<int> v = {1, 2};
    v.resize(5);
    CHECK(v.size() == 5);
    CHECK(v[0] == 1);
    CHECK(v[1] == 2);
    CHECK(v[4] == 0);

    v.resize(1);
    CHECK(v.size() == 1);
    CHECK(v[0] == 1);
}

void test_resize_with_value() {
    learn::Vector<int> v = {1, 2, 3};
    v.resize(6, 8);
    CHECK(v.size() == 6);
    CHECK(v[3] == 8);
    CHECK(v[5] == 8);
}

void test_pop_back_and_clear() {
    learn::Vector<int> v = {1, 2, 3};
    const std::size_t cap = v.capacity();

    v.pop_back();
    CHECK(v.size() == 2);
    CHECK(v[1] == 2);

    v.clear();
    CHECK(v.size() == 0);
    CHECK(v.empty());
    CHECK(v.capacity() == cap);  // capacity preserved
}

void test_const_access() {
    learn::Vector<int> v = {10, 20, 30};
    CHECK(v[1] == 20);
    CHECK(v.data()[2] == 30);
}

void test_push_back_ref() {
    learn::Vector v = {10, 20};
    v.push_back(v[0]);
    CHECK(v[2] == 10);
}

void test_push_back_ref_move() {
    learn::Vector v = {10, 20};
    v.push_back(std::move(v[0]));
    CHECK(v[2] == 10);
}

void test_resize_ref() {
    learn::Vector v = {10, 20};
    v.resize(6, v[0]);
    CHECK(v[5] == 10);
}

// Non-trivial type: catches missing destructor / copy bugs.
struct Counter {
    int id;
    static int alive;

    explicit Counter(int id = 0) : id(id) { ++alive; }
    Counter(const Counter& other) : id(other.id) { ++alive; }
    Counter(Counter&& other) noexcept : id(other.id) { ++alive; }
    Counter& operator=(const Counter& other) {
        id = other.id;
        return *this;
    }
    Counter& operator=(Counter&& other) noexcept {
        id = other.id;
        return *this;
    }
    ~Counter() { --alive; }
};

int Counter::alive = 0;

void test_non_trivial_type() {
    Counter::alive = 0;
    {
        learn::Vector<Counter> v;
        v.push_back(Counter(1));
        v.push_back(Counter(2));
        
        CHECK(v.size() == 2);
        CHECK(Counter::alive == 2);

        learn::Vector<Counter> copy = v;
        CHECK(Counter::alive == 4);

        copy.pop_back();
        CHECK(Counter::alive == 3);
    }
    CHECK(Counter::alive == 0);  // no leaks
}

}  // namespace

int main() {
    std::cout << "Running Vector exercise tests...\n\n";

    run_test("test_default_construct", test_default_construct);
    run_test("test_count_constructor", test_count_constructor);
    run_test("test_count_value_constructor", test_count_value_constructor);
    run_test("test_initializer_list", test_initializer_list);
    run_test("test_push_back_and_reallocation", test_push_back_and_reallocation);
    run_test("test_copy_constructor_deep_copy", test_copy_constructor_deep_copy);
    run_test("test_copy_assignment", test_copy_assignment);
    run_test("test_copy_assignment_self", test_copy_assignment_self);
    run_test("test_move_constructor", test_move_constructor);
    run_test("test_move_assignment", test_move_assignment);
    run_test("test_reserve", test_reserve);
    run_test("test_resize", test_resize);
    run_test("test_resize_with_value", test_resize_with_value);
    run_test("test_pop_back_and_clear", test_pop_back_and_clear);
    run_test("test_const_access", test_const_access);
    run_test("test_non_trivial_type", test_non_trivial_type);
    run_test("test_push_back_ref", test_push_back_ref);
    run_test("test_push_back_ref_move", test_push_back_ref_move);
    run_test("test_resize_ref", test_resize_ref);

    std::cout << "\n" << tests_run << " checks, " << tests_failed << " failed.\n";

    if (tests_failed == 0) {
        std::cout << "All tests passed!\n";
        return 0;
    }

    std::cout << "Some tests failed — keep implementing my_vector.hpp.\n";
    return 1;
}
