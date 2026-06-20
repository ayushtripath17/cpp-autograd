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

namespace learn {

template <typename T>
class Vector {
public:
    // --- Construction / destruction ---

    Vector() : data_(nullptr), size_(0), capacity_(0) {}

    explicit Vector(std::size_t count) : data_(nullptr), size_(0), capacity_(0) {
        // TODO: allocate storage for `count` default-constructed elements.
        // Set size_ and capacity_ accordingly.
        
        size_ = count;
        capacity_ = 2;
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
        // TODO: allocate and fill with `value`.
        size_ = count;
        capacity_ = 2;
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
        // TODO: allocate and copy elements from init.
        size_ = init.size();
        capacity_ = 2;
        while (capacity_ <= size_) {
            capacity_ *= 2;
        }
        void* raw = ::operator new(capacity_ * sizeof(T));
        data_ = static_cast<T*>(raw);
        for (std::size_t i = 0; i < size_; i++) {
            new (data_ + i) T(*(init.begin() + i));
        }
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

    ~Vector() {
        release_resources();
    }

    Vector(const Vector& other) : data_(nullptr), size_(0), capacity_(0) {
        // TODO: deep copy — allocate new storage and copy other's elements.
        size_ = other.size();
        capacity_ = other.capacity();
        void* raw = ::operator new(capacity_ * sizeof(T));
        data_ = static_cast<T*>(raw);

        for (std::size_t i = 0; i < size_; i++) {
            new (data_ + i) T(other.data()[i]);
        }
    }

    Vector& operator=(const Vector& other) {
        // TODO: deep copy. Handle self-assignment (if (this == &other) return *this).
        // Tip: consider destroy + reallocate, or copy-and-swap.
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

    void set_data(T* newdata_) {
        data_ = newdata_;
    }

    void set_size(std::size_t new_size) {
        size_ = new_size;
    }

    void set_capacity(std::size_t new_capacity) {
        capacity_ = new_capacity;
    }

    // Add `noexcept` once implemented.
    Vector(Vector&& other) noexcept 
        : data_(other.data()), size_(other.size()), capacity_(other.capacity()) {
        // TODO: steal other's resources; leave other in a valid empty state.
        other.set_data(nullptr);
        other.set_size(0);
        other.set_capacity(0);
    }

    Vector& operator=(Vector&& other) noexcept {
        // TODO: free current resources, steal from other, leave other empty.
        // Handle self-assignment.
        if (this == &other) {
            return *this;
        }
        release_resources();
        data_ = other.data();
        size_ = other.size();
        capacity_ = other.capacity();

        other.set_data(nullptr);
        other.set_size(0);
        other.set_capacity(0);
        return *this;
    }

    // --- Element access ---

    T& operator[](std::size_t index) {
        // TODO: bounds check optional for learning; tests use valid indices.
        if (index < 0 || index >= size_) {
            throw std::invalid_argument("Invalid index.");
        }

        return *(data_ + index);
    }

    const T& operator[](std::size_t index) const {
        if (index < 0 || index >= size_) {
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
        // TODO: grow or shrink. New elements are default-constructed.
        capacity_ = new_size * 2;
        void* raw = ::operator new(capacity_ * sizeof(T));
        T* temp = static_cast<T*>(raw);
        for (std::size_t i = 0; i < new_size; i++) {
            if (i < size_) {
                new (temp + i) T(std::move(data_[i]));
                data_[i].~T();
            } else {
                new (temp + i) T();
            }
        }

        ::operator delete(data_);
        data_ = temp;
        size_ = new_size;
    }

    void resize(std::size_t new_size, const T& value) {
        // TODO: grow or shrink. New elements are copies of `value`.
        capacity_ = new_size * 2;
        void* raw = ::operator new(capacity_ * sizeof(T));
        T* temp = static_cast<T*>(raw);
        for (std::size_t i = 0; i < new_size; i++) {
            if (i < size_) {
                new (temp + i) T(std::move(data_[i]));
                data_[i].~T();
            } else {
                new (temp + i) T(value);
            }
        }
        ::operator delete(data_);
        data_ = temp;
        size_ = new_size;
    }

    // --- Modifiers ---

    void push_back(const T& value) {
        // TODO: reallocate if size_ == capacity_, then construct new element at end.
        if (capacity_ == 0) {
            reallocate(2);
        }

        new (data_ + size_) T(value);
        size_ += 1;

        if (size_ == capacity_) {
            reallocate(size_ * 2);
        }
    }

    void push_back(T&& value) {
        // TODO: same as above, but move-construct the new element.
        if (capacity_ == 0) {
            reallocate(2);
        }

        new (data_ + size_) T(std::move(value));
        size_ += 1;

        if (size_ == capacity_) {
            reallocate(size_ * 2);
        }
    }

    void pop_back() {
        // TODO: destroy last element and decrement size_. No-op if empty.
        if (size_ == 0) return;

        size_ -= 1;
        data_[size_].~T();

        if (size_ * 4 <= capacity_) {
            capacity_ /= 2;
        }
    }

    void clear() noexcept {
        // TODO: destroy all elements; size_ = 0; keep capacity_ unchanged.
        destroy_elements();
    }

private:
    T* data_;
    std::size_t size_;
    std::size_t capacity_;

    void destroy_elements() noexcept {
        // TODO: call destructors for elements in [0, size_) if needed.
        // For trivial types this may be a no-op, but write it generically
        // using a loop and explicit destructor calls, or std::destroy_n in C++17.
        for (std::size_t i = 0; i < size_; i++) {
            data_[i].~T();
        }
        size_ = 0;
        
    }

    void reallocate(std::size_t new_capacity) {
        // TODO: allocate new block, move existing elements, free old block.
        
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
};

}  // namespace learn
