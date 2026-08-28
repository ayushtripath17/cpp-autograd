#pragma once

// Vector exercise — implement the TODOs below.
//
// Goal: a std::vector-like dynamic array that owns its memory (RAII + Rule of 5).
//
// Suggested order:
//   1. Constructors, destructor, size/capacity/empty, operator[], data()
//   2. push_back, reserve, reallocate (private helper)
//   3. Copy constructor and copy assignment (deep copy!)
//   4. Move constructor and move assignment
//   5. resize, clear, pop_back
//
// Build & run:
//   cmake -S . -B build && cmake --build build && ./build/test_vector

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <new>
#include <optional>

namespace learn {

template <typename T>
class Vector {

private:
    T* data_;
    std::size_t size_;
    std::size_t capacity_;

    void destroy_elements() noexcept {
        for (std::size_t i = 0; i < size_; i++) {
            data_[i].~T();
        }
        size_ = 0; 
    }

    void reallocate(std::size_t new_capacity) {
        // allocate new block, move existing elements, free old block.
        
        void* raw = ::operator new(new_capacity * sizeof(T));
        T* temp = static_cast<T*>(raw);
        for (std::size_t i = 0; i < size_; i++) {
            new (temp + i) T(std::move(data_[i]));
            data_[i].~T();
        }
        ::operator delete(data_);
        data_ = temp;
        capacity_ = new_capacity;
    }

    void release_resources() noexcept {
        for (std::size_t i = 0; i < size_; i++) {
            data_[i].~T();
        }
        ::operator delete(data_);

        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
    }

public:
    // --- Construction / destruction ---

    Vector() : data_(nullptr), size_(0), capacity_(0) {}

    explicit Vector(std::size_t count) : data_(nullptr), size_(0), capacity_(0) {
        // allocate storage for `count` default-constructed elements.
        // set size_ and capacity_ accordingly.
        
        size_ = count;
        capacity_ = 1;
        while (capacity_ <= count) {
            capacity_ *= 2;
        }
        void* raw = ::operator new(capacity_ * sizeof(T));
        data_ = static_cast<T*>(raw);
        for (std::size_t i = 0; i < count; i++) {
            new (data_ + i) T();
        }
    }

    Vector(std::size_t count, const T& value) : data_(nullptr), size_(0), capacity_(0) {
        size_ = count;
        capacity_ = 1;
        while (capacity_ <= count) {
            capacity_ *= 2;
        }
        void* raw = ::operator new(capacity_ * sizeof(T));
        data_ = static_cast<T*>(raw);
        for (std::size_t i = 0; i < count; i++) {
            new (data_ + i) T(value);
        }
    }

    Vector(std::initializer_list<T> init) : data_(nullptr), size_(0), capacity_(0) {
        size_ = init.size();
        capacity_ = 1;
        while (capacity_ <= size_) {
            capacity_ *= 2;
        }
        void* raw = ::operator new(capacity_ * sizeof(T));
        data_ = static_cast<T*>(raw);
        for (std::size_t i = 0; i < size_; i++) {
            new (data_ + i) T(*(init.begin() + i));
        }
    }

    ~Vector() {
        release_resources();
    }

    Vector(const Vector& other) : data_(nullptr), size_(0), capacity_(0) {
        // deep copy — allocates new storage and copy other's elements.

        size_ = other.size();
        capacity_ = other.capacity();
        void* raw = ::operator new(capacity_ * sizeof(T));
        data_ = static_cast<T*>(raw);

        for (std::size_t i = 0; i < size_; i++) {
            new (data_ + i) T(other.data()[i]);
        }
    }

    Vector& operator=(const Vector& other) {
        if (this == &other) {
            return *this;
        }
        
        release_resources();
        void* raw = ::operator new(other.capacity() * sizeof(T));
        data_ = static_cast<T*>(raw);
        size_ = other.size();
        capacity_ = other.capacity();

        for (std::size_t i = 0; i < size_; i++) {
            new (data_ + i) T(other.data()[i]);
        }

        return *this;
    }

    Vector(Vector&& other) noexcept 
        : data_(other.data()), size_(other.size()), capacity_(other.capacity()) {
        // TODO: steal other's resources; leave other in a valid empty state.
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    Vector& operator=(Vector&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        release_resources();
        data_ = other.data_;
        size_ = other.size_;
        capacity_ = other.capacity_;

        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
        return *this;
    }

    // --- Element access ---

    T& operator[](std::size_t index) {
        return *(data_ + index);
    }

    const T& operator[](std::size_t index) const {
        if (index >= size_) {
            throw std::invalid_argument("Invalid index.");
        }

        return *(data_ + index);
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    bool operator==(const Vector& other) const {
        if (size() != other.size()) return false;
        for (std::size_t i = 0; i < size(); i++) {
            if (data_[i] != other.data()[i]) return false;
        }
        return true;
    }

    bool operator!=(const Vector& other) const {
        return !(*(this) == other);
    }

    // --- Capacity ---

    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }

    void reserve(std::size_t new_capacity) {
        // TODO: grow storage if new_capacity > capacity_. Do not change size_.
        if (new_capacity <= capacity_) return;

        this->reallocate(new_capacity);
    }

    void resize(std::size_t new_size) {
        if (size_ == new_size) return;

        if (size_ < new_size) {
            if (capacity_ < new_size) reallocate(new_size * 2);
            for (std::size_t i = size_; i < new_size; i++) {
                new (data_ + i) T();
            }
        } else if (size_ > new_size) {
            for (std::size_t i = new_size; i < size_; i++) {
                data_[i].~T();
            }
        }
        size_ = new_size;
    }

    void resize(std::size_t new_size, const T& value) {
        if (size_ == new_size) return;
        
        if (size_ < new_size) {
            std::optional<T> temp_copy = std::nullopt;
            if (capacity_ < new_size) {
                temp_copy = value;
                reallocate(new_size * 2);
            }
            for (std::size_t i = size_; i < new_size; i++) {
                if (temp_copy.has_value()) new (data_ + i) T(temp_copy.value());
                else new (data_ + i) T(value);
            }
        } else if (size_ > new_size) {
            for (std::size_t i = new_size; i < size_; i++) {
                data_[i].~T();
            }
        }
        size_ = new_size;
    }

    // --- Modifiers ---

    void push_back(const T& value) {
        std::optional<T> temp_copy = std::nullopt;
        if (size_ == capacity_) {
            std::size_t new_size = size_ == 0 ? 2 : size_ * 2;
            temp_copy = value;
            reallocate(new_size);
        }
        if (temp_copy.has_value()) new (data_ + size_) T(temp_copy.value());
        else new (data_ + size_) T(value);
        
        size_ += 1;
    }

    void push_back(T&& value) {
        std::optional<T> temp_copy = std::nullopt;
        if (size_ == capacity_) {
            std::size_t new_size = size_ == 0 ? 2 : size_ * 2;
            temp_copy = std::move(value);
            reallocate(new_size);
        }
        if (temp_copy.has_value()) new (data_ + size_) T(std::move(temp_copy.value()));
        else new (data_ + size_) T(std::move(value));
        
        size_ += 1;
    }

    void pop_back() {
        if (size_ == 0) return;
        size_ -= 1;
        data_[size_].~T();
    }

    void clear() noexcept {
        // destroys all elements; size_ = 0; keeps capacity_ unchanged.
        destroy_elements();
    }

};

}  // namespace learn
