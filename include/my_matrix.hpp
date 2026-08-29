#pragma once

// Matrix exercise — implement the TODOs below.
//
// Goal: a row-major 2D matrix with RAII + Rule of 5, basic linear-algebra ops.
//
// Storage layout (row-major, like NumPy/PyTorch on CPU):
//   index(i, j) = i * cols_ + j
//
// Suggested order:
//   1. Constructors, destructor, rows/cols, operator(), data()
//   2. Element-wise add/subtract, scalar multiply
//   3. transpose
//   4. matmul (matrix multiply)
//   5. Copy constructor and copy assignment (deep copy!)
//   6. Move constructor and move assignment
//
// Build & run:
//   cmake -S . -B build && cmake --build build && ./build/test_matrix

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include "my_vector.hpp"
#include <thread>
#include <vector>

namespace learn {

template <typename T>
class Matrix {
public:
    // --- Construction / destruction ---

    Matrix() : data_(), rows_(0), cols_(0) {}

    Matrix(std::size_t rows, std::size_t cols)
        : data_(rows * cols), rows_(rows), cols_(cols) {
        // TODO: allocate rows * cols storage, default-construct (or zero-fill) elements.
        // Set rows_ and cols_.
    }

    Matrix(std::size_t rows, std::size_t cols, const T& value)
        : data_(rows * cols, value), rows_(rows), cols_(cols) {
        // TODO: allocate and fill every element with `value`.
    }

    Matrix(std::size_t rows, std::size_t cols, std::initializer_list<T> init)
        : data_(init), rows_(rows), cols_(cols) {
        // TODO: allocate rows * cols elements; copy from init in row-major order.
        // Throw if init.size() != rows * cols.
        if (init.size() != rows * cols) {
            throw std::invalid_argument("Invalid init list.");
        }
    }

    ~Matrix() = default;

    Matrix(const Matrix& other) = default;

    Matrix& operator=(const Matrix& other) = default;
    
    Matrix(Matrix&& other) noexcept = default;

    Matrix& operator=(Matrix&& other) noexcept = default;

    // --- Element access ---

    T& operator()(std::size_t row, std::size_t col) {
        // TODO: bounds check optional for learning; tests use valid indices.
        if (row >= rows_ || row < 0 || col >= cols_ || col < 0) {
            throw std::invalid_argument("Invalid indices.");
        }

        return data_[row * cols_ + col];
    }

    const T& operator()(std::size_t row, std::size_t col) const {
        if (row >= rows_ || row < 0 || col >= cols_ || col < 0) {
            throw std::invalid_argument("Invalid indices.");
        }

        return data_[row * cols_ + col];
    }

    Vector<T>& data() noexcept { return data_; }
    const Vector<T>& data() const noexcept { return data_; }

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t size() const noexcept { return rows_ * cols_; }
    bool empty() const noexcept { return data_.empty(); }

    // --- Operations ---

    Matrix operator+(const Matrix& other) const {
        // TODO: element-wise add. Shapes must match.
        if (rows_ != other.rows() || cols_ != other.cols()) {
            throw std::invalid_argument("Dimensions must match");
        }

        Matrix temp(rows_, cols_);
        for (std::size_t i = 0; i < rows_ * cols_; i++) {
            temp.data()[i] = data_[i] + other.data()[i];
        }
        return temp;
    }

    Matrix operator-(const Matrix& other) const {
        // TODO: element-wise subtract. Shapes must match.
        if (rows_ != other.rows() || cols_ != other.cols()) {
            throw std::invalid_argument("Dimensions must match");
        }

        Matrix temp(rows_, cols_);
        for (std::size_t i = 0; i < rows_ * cols_; i++) {
            temp.data()[i] = data_[i] - other.data()[i];
        }
        return temp;
    }

    Matrix operator*(const T& scalar) const {
        // TODO: multiply every element by scalar.

        Matrix temp(rows_, cols_);
        for (std::size_t i = 0; i < rows_ * cols_; i++) {
            temp.data()[i] = data_[i] * scalar;
        }
        return temp;
    }

    Matrix transpose() const {
        // TODO: return cols_ x rows_ matrix with element (i,j) = (*this)(j,i).
        Matrix temp(cols_, rows_);
        
        for (std::size_t i = 0; i < rows_; i++) {
            for (std::size_t j = 0; j < cols_; j++) {
                temp.data()[j * rows_ + i] = data_[i * cols_ + j];
            }
        }

        return temp;
    }

    Matrix matmul(const Matrix& other) const {
        if (this->cols_ != other.rows()) {
            throw std::invalid_argument("Invalid dimensions.");
        }
        Matrix product(rows_, other.cols_, T());
        matmul_into(product, other);
        return product;
    }

    void matmul_into(Matrix& out, const Matrix& other) const {
        if (cols_ != other.rows_ || out.rows_ != rows_ || out.cols_ != other.cols_) {
            throw std::invalid_argument("Invalid dimensions.");
        }

        const std::size_t out_cols = other.cols_;
        for (std::size_t i = 0; i < out.size(); ++i) {
            out.data_[i] = T();
        }

        for (std::size_t i = 0; i < rows_; i++) {
            for (std::size_t j = 0; j < cols_; j++) {
                T local = data_[i * cols_ + j];
                for (std::size_t k = 0; k < out_cols; k++) {
                    out.data_[i * out_cols + k] += local * other.data_[j * out_cols + k];
                }
            }
        }
    }

    Matrix blocked_matmul(const Matrix& other, std::size_t block_size) const {
        if (this->cols_ != other.rows_) {
            throw std::invalid_argument("Invalid dimensions.");
        }

        Matrix product(rows_, other.cols_, T());

        for (std::size_t ii = 0; ii < rows_; ii += block_size) {
            for (std::size_t kk = 0; kk < other.cols_; kk += block_size) {
                for (std::size_t jj = 0; jj < cols_; jj += block_size) {

                    std::size_t row_bound = std::min(rows_, ii + block_size);
                    std::size_t col_bound = std::min(cols_, jj + block_size);
                    std::size_t other_col_bound = std::min(other.cols_, kk + block_size);

                    for (std::size_t i = ii; i < row_bound; i++) {
                        for (std::size_t j = jj; j < col_bound; j++) {
                            const T local = data_[i * cols_ + j];
                            for (std::size_t k = kk; k < other_col_bound; k++) {
                                product.data_[i * other.cols_ + k] += local * other.data_[j * other.cols_ + k];
                            }
                        }
                    }
                }
            }
        }
        return product;
    }

    Matrix multithreaded_matmul(const Matrix& other, std::size_t num_threads) const {
        if (this->cols_ != other.rows_ || num_threads == 0) {
            throw std::invalid_argument("Invalid dimensions/thread count.");
        }

        Matrix product(rows_, other.cols_, T());

        const T* a = data_.data();
        const T* b = other.data_.data();
        T* c = product.data_.data();
        const std::size_t a_cols = cols_;
        const std::size_t b_cols = other.cols_;

        std::vector<std::thread> threads;
        num_threads = std::min(num_threads, rows_);
        for (std::size_t i = 0; i < num_threads; i++) {
            std::size_t begin = (i * rows_) / num_threads;
            std::size_t end = ((i + 1) * rows_) / num_threads;
            
            threads.emplace_back([=] {
                matmul_rows(a, a_cols, b, b_cols, c, begin, end);
            });
        }

        for (std::size_t i = 0; i < num_threads; i++) {
            threads[i].join();
        }
        return product;
    }

private:
    Vector<T> data_;
    std::size_t rows_;
    std::size_t cols_;

    static std::size_t index(std::size_t row, std::size_t col, std::size_t cols) noexcept {
        return row * cols + col;
    }

    static void matmul_rows(
        const T* a, std::size_t a_cols,
        const T* b, std::size_t b_cols,
        T* c,
        std::size_t row_begin, std::size_t row_end)
    {
        for (std::size_t i = row_begin; i < row_end; ++i) {
            for (std::size_t j = 0; j < a_cols; ++j) {
                const T local = a[i * a_cols + j];
                for (std::size_t k = 0; k < b_cols; ++k) {
                    c[i * b_cols + k] += local * b[j * b_cols + k];
                }
            }
        }
    }
};

}  // namespace learn
