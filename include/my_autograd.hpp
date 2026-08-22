#pragma once

// Autograd exercise — implement the TODOs below.
//
// Goal: reverse-mode autograd on Tensor<float> — build a graph on forward,
//       populate .grad on backward, update params with SGD.
//
// Layout convention (match README):
//   x: (batch, in_features)
//   W: (in_features, out_features)
//   b: (out_features,)  — broadcast over batch in add
//   y = matmul(x, W) + b
//
// Suggested order:
//   1. Variable (data, grad, requires_grad, zero_grad)
//   2. add, mul, relu, matmul — forward + register backward_fn
//   3. backward (topological walk from loss)
//   4. mse_loss
//   5. SGD
//   6. Linear, ReLU, Sequential
//
// Graph storage hint:
//   Each op creates a new Variable, stores parent pointers, and sets backward_fn.
//   Keep intermediates in named locals (not dangling temporaries) so parent
//   pointers stay valid until backward().
//
// You will likely want small Tensor helpers (element-wise *, -=, fill zeros).
// Add them here or extend my_tensor.hpp as needed.
//
// Build & run:
//   cmake -S . -B build && cmake --build build && ./build/test_autograd

#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <algorithm>
#include <queue>
#include <random>
#include <unordered_set>
#include <cmath>

#include "my_tensor.hpp"

namespace learn {

using TensorF = Tensor<float>;

// ---------------------------------------------------------------------------
// Variable — one node in the computation graph
// ---------------------------------------------------------------------------

class Variable {
public:
    explicit Variable(TensorF data, bool requires_grad = false)
        : data_(std::move(data)), grad_(), requires_grad_(requires_grad) {
        // TODO: if requires_grad_, initialize grad_ to zeros with same shape as data_.
        if (requires_grad_) {
            grad_ = TensorF(data_.shape());
        }
    }

    TensorF& data() noexcept { return data_; }
    const TensorF& data() const noexcept { return data_; }

    TensorF& grad() noexcept { return grad_; }
    const TensorF& grad() const noexcept { return grad_; }

    bool requires_grad() const noexcept { return requires_grad_; }

    void zero_grad() {
        // TODO: set every element of grad_ to 0.f (keep shape unchanged).
        for (std::size_t i = 0; i < grad_.size(); i++) {
            grad_.data()[i] = 0.f;
        }
    }

private:
    friend std::shared_ptr<Variable> add(std::shared_ptr<Variable>& a, std::shared_ptr<Variable>& b);
    friend std::shared_ptr<Variable> mul(std::shared_ptr<Variable>& a, std::shared_ptr<Variable>& b);
    friend std::shared_ptr<Variable> matmul(std::shared_ptr<Variable>& a, std::shared_ptr<Variable>& b);
    friend std::shared_ptr<Variable> relu(std::shared_ptr<Variable>& x);
    friend std::shared_ptr<Variable> mse_loss(std::shared_ptr<Variable>& pred, Variable& target);
    friend std::shared_ptr<Variable> broadcast_add(std::shared_ptr<Variable>& a, std::shared_ptr<Variable>& b);
    friend std::shared_ptr<Variable> sub(std::shared_ptr<Variable>& a, std::shared_ptr<Variable>& b);
    friend std::shared_ptr<Variable> neg(std::shared_ptr<Variable>& a);
    friend std::shared_ptr<Variable> sum(std::shared_ptr<Variable>& a);
    friend void backward(Variable& loss);
    friend std::vector<Variable*> prims(std::vector<Variable*>& nodes);
    friend std::vector<Variable*> topological_sort(Variable& node);
    friend TensorF broadcast(TensorF& a, TensorF& b);

    TensorF data_;
    TensorF grad_;
    bool requires_grad_ = false;

    // Parents in the forward graph (non-owning — see header comment above).
    std::vector<std::shared_ptr<Variable>> parents_;

    // Called during backward() to propagate loss.grad into parents.
    std::function<void()> backward_fn_;
};

// ---------------------------------------------------------------------------
// Forward ops — each builds a new Variable and wires backward_fn
// ---------------------------------------------------------------------------

inline std::shared_ptr<Variable> add(std::shared_ptr<Variable>& a, std::shared_ptr<Variable>& b) {
    // TODO: forward — out.data = a.data + b.data (element-wise, same shape).
    // TODO: if either input requires grad, set out.requires_grad and record parents.
    // TODO: backward_fn — a.grad += out.grad, b.grad += out.grad (when requires_grad).
    auto out = std::make_shared<Variable>(Variable(a->data() + b->data(), a->requires_grad() || b->requires_grad()));

    if (!out->requires_grad()) return out;
    out->parents_ = {a, b};
    
    out->backward_fn_ = [a_ptr = a.get(), b_ptr = b.get(), out_ptr = out.get()]() {
        if (a_ptr->requires_grad_) a_ptr->grad_ += out_ptr->grad_;
        if (b_ptr->requires_grad_) b_ptr->grad_ += out_ptr->grad_;
    };

    return out;
}

inline std::shared_ptr<Variable> sub(std::shared_ptr<Variable>& a, std::shared_ptr<Variable>& b) {
    
    auto out = std::make_shared<Variable>(Variable(a->data() - b->data(), a->requires_grad() || b->requires_grad()));
    if (!out -> requires_grad()) return out;
    
    out->parents_ = {a, b};
    out->backward_fn_ = [a_ptr = a.get(), b_ptr = b.get(), out_ptr = out.get()]() {
        if (a_ptr->requires_grad()) a_ptr->grad_ += out_ptr->grad_;
        if (b_ptr->requires_grad()) b_ptr->grad_ -= out_ptr->grad_;
    };

    return out;
}

inline std::shared_ptr<Variable> neg(std::shared_ptr<Variable>& a) {
    TensorF n = a->data().shape();
    for (std::size_t i = 0; i < a->data().size(); i++) {
        n.data()[i] = a->data().data()[i] * -1;
    }
    auto out = std::make_shared<Variable>(Variable(n, a->requires_grad()));
    if (!out->requires_grad()) return out;

    out->parents_ = {a};
    out->backward_fn_ = [a_ptr = a.get(), out_ptr = out.get()]() {
        a_ptr->grad_ -= out_ptr->grad_;
    };

    return out;
}

inline std::shared_ptr<Variable> sum(std::shared_ptr<Variable>& a) {
    float total = 0.f;
    for (std::size_t i = 0; i < a->data().size(); i++) {
        total += a->data().data()[i];
    }

    auto out = std::make_shared<Variable>(Variable(TensorF({1}, total), a->requires_grad()));
    if (!out->requires_grad()) return out;

    out->parents_ = {a};
    out->backward_fn_ = [a_ptr = a.get(), out_ptr = out.get()]() {
        for (std::size_t i = 0; i < a_ptr->data().size(); i++) {
            a_ptr->grad().data()[i] += out_ptr->grad().data()[0];
        }
    };

    return out;
}

inline TensorF broadcast(TensorF& a, TensorF& b) {
    TensorF total = (a.shape());
    for (std::size_t i = 0; i < a.size(); i += b.size()) {
        for (std::size_t j = i; j < i + b.size(); j++) {
            total.data()[j] += a.data()[j] + b.data()[j - i];
        }
    }
    return total;
}

inline std::shared_ptr<Variable> broadcast_add(std::shared_ptr<Variable>& a, std::shared_ptr<Variable>& b) {
    TensorF result = broadcast(a->data(), b->data());
    auto out = std::make_shared<Variable>(Variable(result, a->requires_grad() || b->requires_grad()));

    if (!out->requires_grad()) return out;
    
    out->parents_ = {a, b};
    out->backward_fn_ = [a_ptr = a.get(), b_ptr = b.get(), out_ptr = out.get()]() {
        if (a_ptr->requires_grad_) a_ptr->grad_ += out_ptr->grad_;
        if (b_ptr->requires_grad_) {
            for (std::size_t i = 0; i < out_ptr->data().size(); i++) {
                std::size_t col = i % b_ptr->data().size();
                b_ptr->grad_.data()[col] += out_ptr->grad_.data()[i];
            }
        }
    };
    return out;
}

inline std::shared_ptr<Variable> mul(std::shared_ptr<Variable>& a, std::shared_ptr<Variable>& b) {
    // TODO: forward — out.data = a.data * b.data element-wise.
    // TODO: backward — a.grad += out.grad * b.data, b.grad += out.grad * a.data
    //       (use forward values of a/b, not .grad).
    auto out = std::make_shared<Variable>(Variable(a->data_ * b->data_, a->requires_grad() || b->requires_grad()));

    if (!out->requires_grad()) return out;
    
    out->parents_ = {a, b};
    out->backward_fn_ = [a_ptr = a.get(), b_ptr = b.get(), out_ptr = out.get()]() {
        if (a_ptr->requires_grad_) {
            for (std::size_t i = 0; i < a_ptr->grad_.size(); i++) {
                a_ptr->grad_.data()[i] += out_ptr->grad_.data()[i] * b_ptr->data_.data()[i];
            }
        }
        if (b_ptr->requires_grad_) {
            for (std::size_t i = 0; i < b_ptr->grad_.size(); i++) {
                b_ptr->grad_.data()[i] += out_ptr->grad_.data()[i] * a_ptr->data_.data()[i];
            }
        }
    };

    return out;
}

inline std::shared_ptr<Variable> matmul(std::shared_ptr<Variable>& a, std::shared_ptr<Variable>& b) {
    // TODO: forward — out.data = a.data.matmul(b.data).
    // TODO: backward — standard batched matmul gradients using transposes
    //       on the last two dimensions (match your Tensor::matmul convention).
    auto out = std::make_shared<Variable>(Variable(a->data_.matmul(b->data_), a->requires_grad_ || b->requires_grad_));
    if (!out->requires_grad()) return out;

    out->parents_ = {a, b};
    out->backward_fn_ = [a_ptr = a.get(), b_ptr = b.get(), out_ptr = out.get()]() {
        if (a_ptr->requires_grad_) {
            a_ptr->grad_ += out_ptr->grad_.matmul(b_ptr->data_.transpose(b_ptr->data_.ndim() - 2, b_ptr->data_.ndim() - 1));
        }
        if (b_ptr->requires_grad_) {
            b_ptr->grad_ += (a_ptr->data_.transpose(a_ptr->data_.ndim() - 2, a_ptr->data_.ndim() - 1)).matmul(out_ptr->grad_);
        }
    };

    return out;
}

inline TensorF relu_conv(TensorF data) {
    TensorF modified = TensorF(data.shape());
    for (std::size_t i = 0; i < data.size(); i++) {
        modified.data()[i] = std::max(0.f, data.data()[i]);
    }
    return modified;
}

inline std::shared_ptr<Variable> relu(std::shared_ptr<Variable>& x) {
    // TODO: forward — out[i] = max(0, x[i]).
    // TODO: backward — x.grad += out.grad where x.data > 0, else 0.
    auto out = std::make_shared<Variable>(Variable(relu_conv(x->data_), x->requires_grad_));
    if (!out->requires_grad()) return out;

    out->parents_ = {x};
    out->backward_fn_ = [x_ptr = x.get(), out_ptr = out.get()]() {
        for (std::size_t i = 0; i < x_ptr->data_.size(); i++) {
            x_ptr->grad_.data()[i] += x_ptr->data_.data()[i] > 0 ? out_ptr->grad_.data()[i] : 0;
        }
    };

    return out;
}

// new design:
// operations return a shared_ptr to the resulting variable
// variables hold a std::vector<std::shared_ptr<Variable>> containing its parents
// backward_fn_ lambda captures parents by reference
// forward pass for any layer outputs a shared pointer pointing to the resulting value
// graph stays alive as long as this resulting value is in scope;
// as soon as the forward pass value goes out of scope, graph destructs
// prod_ no longer needs to be a member of Linear and Sequential no longer needs to store/return the intermediate resultants

// ---------------------------------------------------------------------------
// Loss
// ---------------------------------------------------------------------------

inline std::shared_ptr<Variable> mse_loss(std::shared_ptr<Variable>& pred, Variable& target) {
    // forward — mean of (pred - target)^2; store as 1-element or scalar tensor.
    // backward w.r.t. pred — 2 * (pred - target) / n  (n = num elements).
    // target is usually a leaf with requires_grad = false.

    if (pred->data_.shape() != target.data_.shape()) throw std::invalid_argument("Prediction and target must have matching shapes.");

    float total = 0.f;
    for (std::size_t i = 0; i < pred->data_.size(); i++) {
        total += (pred->data_.data()[i] - target.data_.data()[i]) * (pred->data_.data()[i] - target.data_.data()[i]);
    }
    float avg = total / pred->data_.size();

    auto out = std::make_shared<Variable>(Variable(TensorF({1}, avg), pred->requires_grad_));
    TensorF target_data = target.data_;
    if (!out->requires_grad_) return out;

    out->parents_ = {pred};
    out->backward_fn_ = [pred_ptr = pred.get(), out_ptr = out.get(), target_data = std::move(target_data)]() {
        for (std::size_t i = 0; i < pred_ptr->data_.size(); i++) {
            pred_ptr->grad_.data()[i] += out_ptr->grad_.data()[0] * (2 * (pred_ptr->data_.data()[i] - target_data.data()[i]) / pred_ptr->data_.size());
        }
    };
    
    return out;
}

// ---------------------------------------------------------------------------
// Backward engine
// ---------------------------------------------------------------------------

inline std::vector<Variable*> prims(std::vector<Variable*>& nodes) {
    std::unordered_map<Variable*, int> in;
    std::vector<Variable*> toposort;

    for (std::size_t i = 0; i < nodes.size(); i++) {
        in[nodes[i]] = 0;
    }
    for (std::size_t i = 0; i < nodes.size(); i++) {
        for (std::size_t j = 0; j < nodes[i]->parents_.size(); j++) {
            in[nodes[i]->parents_[j].get()] += 1;
        }
    }

    std::queue<Variable*> zero;
    for (std::size_t i = 0; i < nodes.size(); i++) {
        if (in[nodes[i]] == 0) {
            zero.push(nodes[i]);
        }
    }

    while (!zero.empty()) {
        Variable* v = zero.front();
        zero.pop();

        toposort.push_back(v);
        for (std::size_t i = 0; i < v->parents_.size(); i++) {
            in[v->parents_[i].get()] -= 1;

            if (in[v->parents_[i].get()] == 0) {
                zero.push(v->parents_[i].get());
            }
        }
    }
    return toposort;
}

inline std::vector<Variable*> topological_sort(Variable& node) {
    std::vector<Variable*> all;
    std::vector<Variable*> stack;
    std::unordered_set<Variable*> seen;

    stack.push_back(&node);

    while (!stack.empty()) {
        Variable* curr = stack.back();
        stack.pop_back();

        if (seen.count(curr)) continue;

        seen.insert(curr);

        all.push_back(curr);

        for (std::size_t i = 0; i < curr->parents_.size(); i++) {
            stack.push_back(curr->parents_[i].get());
        }
    }
    return prims(all);
}

inline void backward(Variable& loss) {
    // TODO:
    //   1. Build topological order from loss (visit parents, no duplicates).
    //   2. Set loss.grad to ones (same shape as loss.data).
    //   3. Walk topo; for each node with backward_fn, call it.
    //   4. Skip nodes where requires_grad() is false.
    // Gradients accumulate (parent.grad += ...); caller should zero_grad first.
    std::vector<Variable*> topo_sorted = topological_sort(loss);
    // for (std::size_t j = 0; j < topo_sorted.size(); j++) {
    //     std::cout << (*topo_sorted[j]).data().data()[0] << "\n";
    // }
    loss.grad_ = TensorF(loss.data_.shape());
    for (std::size_t i = 0; i < loss.grad_.size(); i++) {
        loss.grad_.data()[i] = 1.f;
    }

    for (std::size_t j = 0; j < topo_sorted.size(); j++) {
        if (topo_sorted[j]->backward_fn_) {
            topo_sorted[j]->backward_fn_();
        }
    }
}

// ---------------------------------------------------------------------------
// Optimizer
// ---------------------------------------------------------------------------

class SGD {
    public:
        SGD(std::vector<Variable*> parameters, float lr)
            : parameters_(std::move(parameters)), lr_(lr) {}

        void step() {
            // TODO: for each param with requires_grad:
            //         param->data()[i] -= lr_ * param->grad()[i]
            for (std::size_t i = 0; i < parameters_.size(); i++) {
                if (parameters_[i]->requires_grad()) {
                    for (std::size_t j = 0; j < parameters_[i]->data().size(); j++) {
                        parameters_[i]->data().data()[j] -= lr_ * parameters_[i]->grad().data()[j];
                    }
                }
            }
        }

        void zero_grad() {
            // TODO: call zero_grad() on every parameter in parameters_.
            for (std::size_t i = 0; i < parameters_.size(); i++) {
                parameters_[i]->zero_grad();
            }
        }

    private:
        std::vector<Variable*> parameters_;
        float lr_;
};

class Adam {
    public:
        Adam(std::vector<Variable*> parameters, float lr = 0.001f, 
            float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
            : parameters_(std::move(parameters)), lr_(lr), t_(0), beta1_(beta1), beta2_(beta2), eps_(eps) {
                for (std::size_t i = 0; i < parameters_.size(); i++) {
                    m_.push_back(TensorF(parameters_[i]->data().shape()));
                    v_.push_back(TensorF(parameters_[i]->data().shape()));
                }
            }
        
            void step() {
                t_++;
                for (std::size_t i = 0; i < parameters_.size(); i++) {
                    if (parameters_[i]->requires_grad()) {
                        m_[i] = m_[i] * beta1_ + parameters_[i]->grad() * (1 - beta1_);
                        v_[i] = v_[i] * beta2_ + (parameters_[i]->grad() * parameters_[i]->grad()) * (1 - beta2_);
                        TensorF m_hat = m_[i] / (1 - std::pow(beta1_, t_));
                        TensorF v_hat = v_[i] / (1 - std::pow(beta2_, t_));
                        parameters_[i]->data() -= m_hat * lr_ / (TensorF::sqrt(v_hat) + eps_);
                    }
                }
            }

            void zero_grad() {
                // TODO: call zero_grad() on every parameter in parameters_.
                for (std::size_t i = 0; i < parameters_.size(); i++) {
                    parameters_[i]->zero_grad();
                }
            }

    private:
        std::vector<Variable*> parameters_;
        float lr_;
        std::vector<TensorF> m_;
        std::vector<TensorF> v_;
        int t_;
        float beta1_;
        float beta2_;
        float eps_;
};

// ---------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------

struct LayerType {
    virtual ~LayerType() = default;

    virtual std::shared_ptr<Variable> forward(std::shared_ptr<Variable>& x) = 0;
    virtual std::vector<Variable*> parameters() = 0; 

};

class ReLU : public LayerType {
public:
    std::shared_ptr<Variable> forward(std::shared_ptr<Variable>& x) {
        // TODO: return relu(x).
        return relu(x);
    }

    std::vector<Variable*> parameters() { return {}; }
};

class Linear : public LayerType {
public:
    Linear(std::size_t in_features, std::size_t out_features)
        : W_(std::make_shared<Variable>(TensorF({in_features, out_features}), true)),
          b_(std::make_shared<Variable>(TensorF({out_features}), true)) {
        
        double limit = sqrt(6 / (in_features + out_features));
        std::random_device rd; 
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distr(-1 * limit, limit);
        for (std::size_t i = 0; i < W_->data().size(); i++) {
            W_->data().data()[i] = distr(gen);
        }
    }

    Linear(std::size_t in_features, std::size_t out_features, std::mt19937& rng)
        : W_(std::make_shared<Variable>(TensorF({in_features, out_features}), true)),
          b_(std::make_shared<Variable>(TensorF({out_features}), true)) {
        
        double limit = sqrt(6 / (in_features + out_features));
        std::uniform_real_distribution<float> distr(-1 * limit, limit);
        for (std::size_t i = 0; i < W_->data().size(); i++) {
            W_->data().data()[i] = distr(rng);
        }
    }

    std::shared_ptr<Variable> forward(std::shared_ptr<Variable>& x) {
        // TODO: return add(matmul(x, W_), b_) with broadcasting on b.
        
        std::shared_ptr<Variable> prod = matmul(x, W_);
        std::shared_ptr<Variable> result = broadcast_add(prod, b_);
        return result;
    }

    std::vector<Variable*> parameters() {
        // TODO: return {&W_, &b_}.
        return {W_.get(), b_.get()};
    }

private:
    std::shared_ptr<Variable> W_;
    std::shared_ptr<Variable> b_;
};

class Sequential {
public:

    template <typename LayerT>
    void add(LayerT layer) {
        // TODO: store layer for forward/parameters.
        layers_.push_back(std::make_unique<LayerT>(std::move(layer)));
    }

    std::shared_ptr<Variable> forward(Variable& x) {
        // TODO: x -> layer0 -> layer1 -> ...
        auto result = std::make_shared<Variable>(x);
        for (std::size_t i = 0; i < layers_.size(); i++) {
            result = layers_[i]->forward(result);
        }
        return result;
    }

    std::vector<Variable*> parameters() {
        // TODO: concatenate parameters() from all layers.
        std::vector<Variable*> p;
        for (std::size_t i = 0; i < layers_.size(); i++) {
            std::vector<Variable*> temp = layers_[i]->parameters();
            for (const auto& ptr : temp) {
                p.push_back(ptr);
            }
        }
        return p;
    }

private:
    std::vector<std::unique_ptr<LayerType>> layers_;
};

}  // namespace learn
