#ifndef EASY_LLM_LOADER_HPP
#define EASY_LLM_LOADER_HPP

#include <array>
#include <vector>
#include <memory>
#include <unordered_map>

#include "tensor.hpp"
#include "config.hpp"

namespace easy_llm {

class ModelParam {
public:
    static std::unique_ptr<ModelParam> load(const std::string& model_path);
    ModelParam();
    Tensor take_param(const std::string& key);
    const Tensor& peek_param(const std::string& key) const;
    bool contains(const std::string& key) const;

private:
    void load_from_ckpt(const std::string& path);

    std::unordered_map<std::string, Tensor> params_;
};

}  // namespace easy_llm

#endif  // EASY_LLM_LOADER_HPP
