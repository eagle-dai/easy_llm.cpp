#include "tensor.hpp"
#include <cmath>

#include "spdlog/spdlog.h"

namespace easy_llm {

using std::move;
using std::vector;

Tensor::Tensor() {
}

Tensor::Tensor(int size) {
    data_.resize(size);
}

Tensor::Tensor(vector<data_type>& data) : data_(move(data)) {
}

Tensor::Tensor(int size, vector<int> shape) {
    data_.resize(size);
    shape_ = shape;
}

Tensor::Tensor(vector<data_type>& data, vector<int> shape) : data_(move(data)), shape_(shape) {
}

void Tensor::reserve(int new_cap) {
    data_.reserve(new_cap);
}

void Tensor::resize(int new_size) {
    data_.resize(new_size);
}

void Tensor::emplace_back(data_type value) {
    data_.emplace_back(value);
}

Tensor& Tensor::reshape(vector<int> new_shape) {
    shape_ = new_shape;
    int total_size = 1;
    for(auto dim : shape_) {
        total_size *= dim;
    }
    if(total_size != static_cast<int>(data_.size())) {
        for(auto dim : shape_) {
            spdlog::error("{}", dim);
        }
        throw std::invalid_argument("Total size mismatch in reshape.");
    }
    return *this;
}

Tensor& Tensor::transpose(int dim0, int dim1) {
    int rank = static_cast<int>(shape_.size());
    if (dim0 < 0 || dim1 < 0 || dim0 >= rank || dim1 >= rank) {
        throw std::out_of_range("Dimension index out of range.");
    }
    vector<int> old_shape = shape_;
    vector<int> old_strides(old_shape.size(), 1);
    for (int i = rank - 2; i >= 0; --i) {
        old_strides[i] = old_strides[i + 1] * old_shape[i + 1];
    }
    std::swap(shape_[dim0], shape_[dim1]);
    vector<int> new_strides(shape_.size(), 1);
    for (int i = rank - 2; i >= 0; --i) {
        new_strides[i] = new_strides[i + 1] * shape_[i + 1];
    }
    auto original_data = data_;
    data_.resize(original_data.size());
    int total_size = original_data.size();

#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int old_idx = 0; old_idx < total_size; ++old_idx) {
        int remaining = old_idx;
        vector<int> multi_idx(old_shape.size(), 0);
        for (int i = 0; i < rank; ++i) {
            multi_idx[i] = remaining / old_strides[i];
            remaining %= old_strides[i];
        }
        std::swap(multi_idx[dim0], multi_idx[dim1]);
        int new_idx = 0;
        for (int i = 0; i < rank; ++i) {
            new_idx += multi_idx[i] * new_strides[i];
        }
        data_[new_idx] = original_data[old_idx];
    }
    return *this;
}

void Tensor::scale_inplace(float scale) {
    for (data_type& val : data_) {
        val = Bf16(float(val) * scale);
    }
}

Tensor& Tensor::split_head(int num_heads) {
    if (shape_.empty()) {
        throw std::invalid_argument("Tensor has no dimensions to split.");
    }
    int hidden_dim = shape_.back();
    if (hidden_dim % num_heads != 0) {
        throw std::invalid_argument("hidden_dim is not divisible by num_heads.");
    }
    int hidden_dim_per_head = hidden_dim / num_heads;
    shape_.pop_back();
    shape_.push_back(num_heads);
    shape_.push_back(hidden_dim_per_head);
    return *this;
}

Tensor Tensor::softmax() const {
    Tensor output(this->size(), this->shape_);
    int last_dim = shape_.back();
    int outer_size = data_.size() / last_dim;
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for(int i = 0; i < outer_size; ++i){
        data_type max_val = -std::numeric_limits<data_type>::infinity();
        for(int j = 0; j < last_dim; ++j){
            if(data_[i * last_dim + j] > max_val){
                max_val = data_[i * last_dim + j];
            }
        }
        auto sum = data_type(0);
        for(int j = 0; j < last_dim; ++j){
            output.data_[i * last_dim + j] = data_type(::exp(data_[i * last_dim + j] - max_val));
            sum += output.data_[i * last_dim + j];
        }
        const data_type epsilon = data_type(std::numeric_limits<float>::epsilon() * 10.0f);
        sum = std::max(sum, epsilon);
        for(int j = 0; j < last_dim; ++j){
            output.data_[i * last_dim + j] /= sum;
        }
    }
    return output;
}

Tensor& Tensor::gelu() {
    int last_dim = shape_.back();
    int outer_size = data_.size() / last_dim;
#ifdef USE_OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < outer_size; ++i) {
        for (int j = 0; j < last_dim; ++j) {
            int idx = i * last_dim + j;
            data_[idx] = data_[idx] * data_type(0.5f) * (data_type(1.0f) + data_type(::erf(data_[idx]) / std::sqrt(data_type(2.0f))));
        }
    }
    return *this;
}

Tensor& Tensor::silu() {
    int last_dim  = shape_.back();
    int outer_size = data_.size() / last_dim;
#ifdef USE_OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < outer_size; ++i) {
        for (int j = 0; j < last_dim; ++j) {
            int idx = i * last_dim + j;
            data_type x = data_[idx];
            data_type sig = data_type(1.0f) / (data_type(1.0f) + data_type(std::exp(-x)));
            data_[idx] = x * sig;
        }
    }
    return *this;
}

Tensor& Tensor::layernorm() {  // Pure normalization with no parameters; deprecated and only used for parameter-free unit tests
    if (shape_.empty()) {
        throw std::invalid_argument("Cannot apply layernorm to a tensor with no dimensions.");
    }
    int features = shape_.back();
    auto epsilon = data_type(1e-5f);
    int num_rows = data_.size() / features;
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int r = 0; r < num_rows; ++r) {
        size_t i = r * features;
        auto mean = this->mean(i, features);
        auto variance = this->variance(i, features, mean);
        auto stddev = std::sqrt(variance + epsilon);
        for (int j = 0; j < features; ++j) {
            data_[i + j] = (data_[i + j] - mean) / data_type(stddev);
        }
    }
    return *this;
}

data_type Tensor::mean(int start, int length) const {
    data_type mean = data_type(.0f);
    for(int i = start; i < start + length; ++i) {
        mean += data_[i];
    }
    mean = data_type(mean / length);  // mean and length are implicitly cast to float before converting back to bf16
    return mean;
}

data_type Tensor::variance(int start, int length, data_type mean) const {
    data_type variance = data_type(0.0f);
    for(int i = start; i < start + length; ++i) {
        data_type diff = data_[i] - mean;
        variance += diff * diff;
    }
    variance = data_type(variance / length);
    return variance;
}

Tensor& Tensor::repeat(int repeats, int axis) {
    int rank = static_cast<int>(shape_.size());
    if (axis < 0 || axis >= rank) {
        spdlog::error("axis == {}, shape_.size() == {}", axis, shape_.size());
        throw std::out_of_range("Axis index out of range.");
    }
    auto new_shape = shape_;
    new_shape[axis] *= repeats;
    int new_total_size = 1;
    for (int dim : new_shape) new_total_size *= dim;

    int new_rank = static_cast<int>(new_shape.size());
    vector<int> new_strides(new_shape.size(), 1);
    for (int i = new_rank - 2; i >= 0; --i) {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }
    vector<int> old_strides(shape_.size(), 1);
    for (int i = rank - 2; i >= 0; --i) {
        old_strides[i] = old_strides[i + 1] * shape_[i + 1];
    }
    vector<data_type> new_data(new_total_size);

#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < new_total_size; ++i) {
        int temp_idx = i;
        int old_idx = 0;
        
        for (int dim = 0; dim < new_rank; ++dim) {
            int coord = temp_idx / new_strides[dim];
            temp_idx %= new_strides[dim];
            int old_coord = coord;
            if (dim == axis) {
                old_coord = coord / repeats;
            }
            old_idx += old_coord * old_strides[dim];
        }
        new_data[i] = data_[old_idx];
    }

    data_ = move(new_data);
    shape_ = move(new_shape);
    return *this;
}

} // namespace easy_llm
