#ifndef EASY_GPT_TENSOR_HPP
#define EASY_GPT_TENSOR_HPP

#include <vector>

#include "bf16.hpp"

namespace easy_gpt {

#ifdef USE_FP32
    using data_type = float;
#elif defined USE_FP16
    using data_type = float16_t;
#elif defined USE_BF16
    using data_type = Bf16;
#else
    #error "No valid precision defined!"
#endif

class Tensor {
public:
    Tensor();
    Tensor(int size);
    Tensor(std::vector<data_type>& data);
    Tensor(int size, std::vector<int> shape);
    Tensor(std::vector<data_type>& data, std::vector<int> shape);

    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    void reserve(int new_cap);
    void resize(int new_size);
    void emplace_back(data_type value);
    data_type& at(int index) { return data_.at(index); }
    const data_type& at(int index) const { return data_.at(index); }
    data_type& operator[](int index) { return data_[index]; }
    const data_type& operator[](int index) const { return data_[index]; }
    int size() const { return data_.size(); }
    std::vector<int> shape() const { return shape_; }
    std::vector<data_type>& data() { return data_; }
    const std::vector<data_type>& data() const { return data_; }
    
    Tensor& reshape(std::vector<int> new_shape);
    Tensor& transpose(int dim0, int dim1);
    Tensor& split_head(int num_heads);
    void scale_inplace(float scale);
    Tensor softmax() const;
    Tensor& gelu();
    Tensor& silu();
    Tensor& layernorm();
    Tensor& repeat(int repeats, int axis);
    data_type mean(int start, int length) const;
    data_type variance(int start, int length, data_type mean) const;

private:
    std::vector<data_type> data_;
    std::vector<int> shape_;
};


} // namespace easy_gpt

#endif // EASY_GPT_TENSOR_HPP
