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
        // TODO: matrix multiply. (*this) is (m x k), other is (k x n), result is (m x n).
        // C(i,j) = sum_r A(i,r) * B(r,j)
        if (this->cols_ != other.rows()) {
            throw std::invalid_argument("Invalid dimensions.");
        }

        Matrix product(rows_, other.cols());
        
        for (std::size_t i = 0; i < rows_; i++) {
            for (std::size_t z = 0; z < other.cols(); z++) {
                T sum = T();
                for (std::size_t j = 0; j < cols_; j++) {
                    sum += (data_[i * cols_ + j] * other.data()[j * other.cols() + z]);
                }
                product.data()[i * product.cols() + z] = std::move(sum);
            }
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
};

}  // namespace learn
