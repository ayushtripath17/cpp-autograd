#pragma once

// Tensor exercise — implement the TODOs below.
//
// Goal: N-dimensional row-major tensor with shape, strides, views, and batched matmul.
//
// Storage: one contiguous Vector<T> with row-major strides (like NumPy).
//   linear_index = offset + sum_k index[k] * stride[k]
//
// Suggested order:
//   1. Constructors, shape/strides, at() / operator()
//   2. reshape (same numel, contiguous)
//   3. transpose (swap two axes)
//   4. TensorView — non-owning view into parent storage
//   5. element-wise add (same shape)
//   6. matmul — contract last axis of A with second-to-last of B;
//      batch dimensions must match (see tests)
//
// Build & run:
//   cmake -S . -B build && cmake --build build && ./build/test_tensor

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <limits>

#include "my_vector.hpp"
#include "my_matrix.hpp"

namespace learn {

template <typename T>
class Tensor;

// Non-owning view. Parent Tensor must outlive the view.
template <typename T>
class TensorView {
public:
    TensorView() = default;

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    std::size_t ndim() const noexcept { return shape_.size(); }
    std::size_t size() const noexcept { return size_; }
    const Vector<std::size_t>& shape() const noexcept { return shape_; }
    const Vector<std::size_t>& strides() const noexcept { return strides_; }
    std::size_t offset() const noexcept { return offset_; }

    T& at(std::initializer_list<std::size_t> indices) {
        return data_[linear_index(indices)];
    }

    const T& at(std::initializer_list<std::size_t> indices) const {
        return data_[linear_index(indices)];
    }

private:
    friend class Tensor<T>;

    T* data_ = nullptr;
    Vector<std::size_t> shape_;
    Vector<std::size_t> strides_;
    std::size_t offset_ = 0;
    std::size_t size_ = 0;

    std::size_t linear_index(std::initializer_list<std::size_t> indices) const {
        // TODO: validate indices.size() == ndim(), bounds-check each index.
        if (indices.size() != ndim()) {
            throw std::invalid_argument("Invalid shape.");
        }
        std::size_t ind = 0;
        for (std::size_t i = 0; i < indices.size(); i++) {
            if (*(indices.begin() + i) >= shape_[i]) {
                throw std::invalid_argument("Invalid indices.");
            }
            ind += (*(indices.begin() + i) * strides_[i]);
        }
        ind += offset_;
        return ind;
    }
};

template <typename T>
class Tensor {
public:
    // --- Construction ---

    Tensor() = default;

    explicit Tensor(std::initializer_list<std::size_t> shape)
        : data_(), shape_(), strides_(), offset_(0) {
        // TODO: compute numel from shape, allocate data_, fill with T().
        std::size_t total = 1;
        for (std::size_t i = 0; i < shape.size(); i++) {
            total *= *(shape.begin() + i);
        }
        data_ = Vector<T>(total);
        shape_ = Vector<std::size_t>(shape);
        strides_ = Vector<std::size_t>(shape.size(), std::size_t{1});
        for (std::size_t i = 0; i < shape.size(); i++) {
            total /= shape_[i];
            strides_[i] = total;
        }
    }

    Tensor(Vector<std::size_t> shape)
        : data_(), shape_(), strides_(), offset_(0) {
        // TODO: compute numel from shape, allocate data_, fill with T().
        std::size_t total = 1;
        for (std::size_t i = 0; i < shape.size(); i++) {
            total *= shape[i];
        }
        data_ = Vector<T>(total);
        shape_ = shape;
        strides_ = Vector<std::size_t>(shape.size(), std::size_t{1});
        for (std::size_t i = 0; i < shape.size(); i++) {
            total /= shape_[i];
            strides_[i] = total;
        }
    }

    Tensor(std::initializer_list<std::size_t> shape, const T& value)
        : data_(), shape_(), strides_(), offset_(0) {
        // TODO: allocate and fill with value.
        std::size_t total = 1;
        for (std::size_t i = 0; i < shape.size(); i++) {
            total *= *(shape.begin() + i);
        }
        data_ = Vector<T>(total, value);

        shape_ = Vector<std::size_t>(shape);
        strides_ = Vector<std::size_t>(shape.size(), std::size_t{1});
        for (std::size_t i = 0; i < shape.size(); i++) {
            total /= shape_[i];
            strides_[i] = total;
        }
    }

    Tensor(std::initializer_list<std::size_t> shape,
           std::initializer_list<T> values)
        : data_(), shape_(), strides_(), offset_(0) {
        // TODO: values.size() must equal product(shape).
        std::size_t total = 1;
        for (std::size_t i = 0; i < shape.size(); i++) {
            total *= *(shape.begin() + i);
        }
        if (values.size() != total) {
            throw std::invalid_argument("Invalid size of values.");
        }
        
        data_ = Vector(values);
        shape_ = Vector(shape);
        strides_ = Vector<std::size_t>(shape.size(), std::size_t{1});
        for (std::size_t i = 0; i < shape.size(); i++) {
            total /= shape_[i];
            strides_[i] = total;
        }
    }

    // --- Metadata ---

    std::size_t ndim() const noexcept { return shape_.size(); }
    std::size_t size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return size() == 0; }

    const Vector<std::size_t>& shape() const noexcept { return shape_; }
    const Vector<std::size_t>& strides() const noexcept { return strides_; }

    Vector<T>& data() noexcept { return data_; }
    const Vector<T>& data() const noexcept { return data_; }

    // --- Element access ---

    T& at(std::initializer_list<std::size_t> indices) {
        return data_[linear_index(indices)];
    }

    const T& at(std::initializer_list<std::size_t> indices) const {
        return data_[linear_index(indices)];
    }

    // --- Shape ops ---

    Tensor reshape(std::initializer_list<std::size_t> new_shape) const {
        // TODO: product(new_shape) must equal size(). New Tensor shares semantics:
        // copy data into new contiguous Tensor with new shape/strides.
        std::size_t total = 1;
        for (std::size_t i = 0; i < new_shape.size(); i++) {
            total *= *(new_shape.begin() + i);
        }

        if (total != size()) throw std::invalid_argument("Invalid new shape.");

        Tensor reshaped = Tensor<T>(new_shape);
        reshaped.data() = data_;
        return reshaped;
    }

    Tensor transpose(std::size_t axis0, std::size_t axis1) const {
        // TODO: swap those axes in shape/strides; copy elements to new layout.
        if (axis0 < 0 || axis0 >= ndim() || axis1 < 0 || axis1 >= ndim()) {
            throw std::invalid_argument("Invalid indices.");
        }

        Tensor transposed = Tensor<T>();
        Vector<std::size_t> new_shape = Vector(shape_);
        std::swap(new_shape[axis0], new_shape[axis1]);

        transposed.shape_ = new_shape;
        transposed.strides_ = make_strides(new_shape);
        transposed.data_ = Vector<T>(size());
        
        for (std::size_t i = 0; i < size(); i++) {
            Vector<std::size_t> indices = reverse_index(i, this->strides_);
            std::swap(indices[axis0], indices[axis1]);
            transposed.data()[linear_index(indices, transposed.strides_)] = data_[i];
        }

        return transposed;
    }

    TensorView<T> view(std::initializer_list<std::size_t> new_shape) {
        // TODO: product(new_shape) == size(); build view sharing data_.data().
        std::size_t product = 1;
        for (std::size_t i = 0; i < new_shape.size(); i++) {
            product *= *(new_shape.begin() + i);
        }
        if (product != size()) throw std::invalid_argument("Size must remain the same.");
        
        TensorView<T> reshaped = TensorView<T>();
        reshaped.data_ = data_.data();
        reshaped.size_ = data_.size();
        reshaped.shape_ = Vector<std::size_t>(new_shape);
        reshaped.strides_ = make_strides(reshaped.shape_);
        reshaped.offset_ = offset_;

        return reshaped;
    }

    // --- Element-wise ---

    Tensor operator+(const Tensor& other) const {
        // TODO: same shape required.
        if (shape_ != other.shape()) {
            throw std::invalid_argument("Tensors must have the same shape.");
        }
        Tensor sum = Tensor<T>();
        sum.shape_ = this->shape_;
        sum.strides_ = this->strides_;
        sum.data_ = Vector<T>(size());
        for (std::size_t i = 0; i < data_.size(); i++) {
            sum.data()[i] = data_[i] + other.data()[i];
        }
        return sum;
    }

    Tensor operator+(const T& other) const {
        Tensor sum = Tensor<T>();
        sum.shape_ = this->shape_;
        sum.strides_ = this->strides_;
        sum.data_ = Vector<T>(size());
        for (std::size_t i = 0; i < data_.size(); i++) {
            sum.data()[i] = data_[i] + other;
        }
        return sum;
    }

    Tensor operator-(const Tensor& other) const {
        // TODO: same shape required.
        if (shape_ != other.shape()) {
            throw std::invalid_argument("Tensors must have the same shape.");
        }
        Tensor diff = Tensor<T>();
        diff.shape_ = this->shape_;
        diff.strides_ = this->strides_;
        diff.data_ = Vector<T>(size());
        for (std::size_t i = 0; i < data_.size(); i++) {
            diff.data()[i] = data_[i] - other.data()[i];
        }
        return diff;
    }

    Tensor operator-(const T& other) const {
        Tensor diff = Tensor<T>();
        diff.shape_ = this->shape_;
        diff.strides_ = this->strides_;
        diff.data_ = Vector<T>(size());
        for (std::size_t i = 0; i < data_.size(); i++) {
            diff.data()[i] = data_[i] - other;
        }
        return diff;
    }

    void operator+=(const Tensor& other) {
        if (shape_ != other.shape()) {
            throw std::invalid_argument("Tensors must have the same shape.");
        }

        for (std::size_t i = 0; i < data_.size(); i++) {
            this->data_[i] = data_[i] + other.data()[i];
        }
    } 

    void operator-=(const Tensor& other) {
        if (shape_ != other.shape()) {
            throw std::invalid_argument("Tensors must have the same shape.");
        }

        for (std::size_t i = 0; i < data_.size(); i++) {
            this->data_[i] = data_[i] - other.data()[i];
        }
    } 

    Tensor operator*(const Tensor& other) const {
        // TODO: same shape required.
        if (shape_ != other.shape()) {
            throw std::invalid_argument("Tensors must have the same shape.");
        }
        Tensor prod = Tensor<T>();
        prod.shape_ = this->shape_;
        prod.strides_ = this->strides_;
        prod.data_ = Vector<T>(size());
        for (std::size_t i = 0; i < data_.size(); i++) {
            prod.data()[i] = data_[i] * other.data()[i];
        }
        return prod;
    }

    Tensor operator*(const T& other) const {
        Tensor prod = Tensor<T>();
        prod.shape_ = this->shape_;
        prod.strides_ = this->strides_;
        prod.data_ = Vector<T>(size());
        for (std::size_t i = 0; i < data_.size(); i++) {
            prod.data()[i] = data_[i] * other;
        }
        return prod;
    }

    Tensor operator/(const Tensor& other) const {
        // TODO: same shape required.
        if (shape_ != other.shape()) {
            throw std::invalid_argument("Tensors must have the same shape.");
        }
        Tensor q = Tensor<T>();
        q.shape_ = this->shape_;
        q.strides_ = this->strides_;
        q.data_ = Vector<T>(size());
        for (std::size_t i = 0; i < data_.size(); i++) {
            q.data()[i] = data_[i] / other.data()[i];
        }
        return q;
    }

    Tensor operator/(const T& other) const {
        Tensor q = Tensor<T>();
        q.shape_ = this->shape_;
        q.strides_ = this->strides_;
        q.data_ = Vector<T>(size());
        for (std::size_t i = 0; i < data_.size(); i++) {
            q.data()[i] = data_[i] / other;
        }
        return q;
    }

    static Tensor sqrt(const Tensor& n) {
        Tensor q = Tensor<T>();
        q.shape_ = n.shape_;
        q.strides_ = n.strides_;
        q.data_ = Vector<T>(n.size());

        for (std::size_t i = 0; i < q.data_.size(); i++) {
            q.data_[i] = std::pow(n.data_[i], 0.5);
        }
        return q;
    }

    static Tensor exp(const Tensor& n) {
        Tensor e = Tensor<T>(n.shape());
        for (std::size_t i = 0; i < e.size(); i++) {
            e.data_[i] = std::exp(n.data_[i]);
        }
        return e;
    }

    static Tensor log(const Tensor& n) {
        Tensor l = Tensor<T>(n.shape());
        for (std::size_t i = 0; i < l.size(); i++) {
            l.data_[i] = std::log(n.data_[i]);
        }
        return l;
    }

    static Tensor softmax(const Tensor& n) {
        Tensor output = Tensor<T>(n.shape());
        std::size_t last = n.strides_[n.ndim() - 1];
        for (std::size_t i = 0; i < output.size(); i += last) {
            int row_s = 0;
            int maximum = std::numeric_limits<int>::min();
            for (std::size_t j = 0; j < i + last; j++) {
                maximum = std::max(maximum, n.data_[j]);
            }
            for (std::size_t j = i; j < i + last; j++) {
                output.data_[j] = std::exp(n.data_[j] - maximum);
                row_s += output.data_[j];
            }
            for (std::size_t j = i; j < i + last; j++) {
                output.data_[j] /= row_s;
            }
        }
        return output;
    }


    // --- Batched matmul (important for DL) ---
    //
    // Treat trailing dimensions as matrices:
    //   A: (..., m, k),  B: (..., k, n)  ->  C: (..., m, n)
    // Leading "..." must match on A and B (same batch shape).
    //
    // 2D case: shape (m,k) @ (k,n) -> (m,n) — same as Matrix::matmul.
    Tensor matmul(const Tensor& other) const {
        if (ndim() != other.ndim()) {
            throw std::invalid_argument("Number of dimensions must match.");
        }
        if (ndim() == 1 || other.ndim() == 1) {
            throw std::invalid_argument("Can not call matmul on 1d tensor.");
        }

        Vector<std::size_t> new_shape = Vector<std::size_t>();
        for (std::size_t i = 0; i < ndim() - 2; i++) {
            if (shape_[i] != other.shape()[i]) {
                throw std::invalid_argument("Batch dimensions must match.");
            }
            new_shape.push_back(shape_[i]);
        }
        new_shape.push_back(shape_[ndim() - 2]); 
        new_shape.push_back(other.shape()[ndim() - 1]);

        std::size_t total1 = shape_[ndim() - 2] * shape_[ndim() - 1];
        std::size_t total2 = other.shape()[ndim() - 2] * other.shape()[ndim() - 1];

        std::size_t inc2 = 0;
        std::size_t inc3 = 0;
        Tensor result = Tensor(new_shape);
        for (std::size_t inc1 = 0; inc1 < size(); inc1 += total1) {
            Matrix m1 = Matrix<T>(shape_[ndim() - 2], shape_[ndim() - 1]);
            Matrix m2 = Matrix<T>(other.shape()[ndim() - 2], other.shape()[ndim() - 1]);

            for (std::size_t i = inc1; i < (inc1 + total1); i++) {
                m1.data()[i - inc1] = data_[i];
            }
            for (std::size_t i = inc2; i < (inc2 + total2); i++) {
                m2.data()[i - inc2] = other.data()[i];
            }

            Matrix product = m1.matmul(m2);
            for (std::size_t a = 0; a < product.size(); a++) {
                result.data()[inc3 + a] = product.data()[a];
            }
            inc2 += total2;
            inc3 += product.size();
        }

        return result;
    }

private:
    Vector<T> data_;
    Vector<std::size_t> shape_;
    Vector<std::size_t> strides_;
    std::size_t offset_ = 0;

    static std::size_t product(std::initializer_list<std::size_t> shape) {
        std::size_t n = 1;
        for (std::size_t d : shape) {
            if (d == 0) return 0;
            n *= d;
        }
        return n;
    }

    static Vector<std::size_t> make_strides(const Vector<std::size_t>& shape) {
        // TODO: row-major contiguous strides.
        // e.g. shape [2,3,4] -> strides [12, 4, 1]
        std::size_t total = 1;
        for (std::size_t i = 0; i < shape.size(); i++) {
            total *= shape[i];
        }

        Vector stride = Vector<std::size_t>(shape.size(), std::size_t{1});
        for (std::size_t i = 0; i < shape.size(); i++) {
            total /= shape[i];
            stride[i] = total;
        }
        return stride;
    }

    std::size_t linear_index(std::initializer_list<std::size_t> indices) const {
        // TODO: offset_ + sum indices[i] * strides_[i]
        std::size_t ind = 0;
        for (std::size_t i = 0; i < indices.size(); i++) {
            ind += (*(indices.begin() + i) * strides_[i]);
        }
        ind += offset_;
        return ind;
    }

    std::size_t linear_index(Vector<std::size_t>& indices, Vector<std::size_t>& strides) const {
        // TODO: offset_ + sum indices[i] * strides_[i]
        std::size_t ind = 0;
        for (std::size_t i = 0; i < indices.size(); i++) {
            ind += (indices[i] * strides[i]);
        }
        return ind;
    }

    Vector<std::size_t> reverse_index(std::size_t ind, const Vector<std::size_t>& strides) const {
        Vector<std::size_t> indices;
        for (std::size_t i = 0; i < strides_.size(); i++) {
            indices.push_back(ind / strides[i]);
            ind %= strides[i];
        }
        return indices;
    }
};

}  // namespace learn
