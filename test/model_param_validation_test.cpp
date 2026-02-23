#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

#include "config.hpp"
#include "models/layer_key_prefix.hpp"
#include "models/loader.hpp"
#include "models/model_param_validation.hpp"
#include "tensor.hpp"

namespace {

struct TensorSpec {
    std::string key;
    std::vector<int> shape;
    float fill_value;
};

std::vector<easy_llm::data_type> build_tensor_data(const std::vector<int>& shape, float fill_value) {
    int total = 1;
    for (int dim : shape) {
        total *= dim;
    }
    std::vector<easy_llm::data_type> data;
    data.reserve(total);
    for (int i = 0; i < total; ++i) {
        data.emplace_back(easy_llm::data_type(fill_value + static_cast<float>(i) * 0.001f));
    }
    return data;
}

std::filesystem::path write_safetensors_fixture(const std::string& filename,
                                                const std::vector<TensorSpec>& specs) {
    nlohmann::json header = nlohmann::json::object();
    std::vector<char> binary_blob;
    std::uint64_t data_offset = 0;

    for (const auto& spec : specs) {
        std::vector<easy_llm::data_type> data = build_tensor_data(spec.shape, spec.fill_value);
        std::size_t bytes = data.size() * sizeof(easy_llm::data_type);

        std::uint64_t begin = data_offset;
        std::uint64_t end = data_offset + static_cast<std::uint64_t>(bytes);
        data_offset = end;

        header[spec.key] = {
            {"dtype", "BF16"},
            {"shape", spec.shape},
            {"data_offsets", {begin, end}}
        };

        const char* ptr = reinterpret_cast<const char*>(data.data());
        binary_blob.insert(binary_blob.end(), ptr, ptr + bytes);
    }

    std::string header_str = header.dump();
    std::uint64_t header_len = static_cast<std::uint64_t>(header_str.size());

    std::filesystem::path path = std::filesystem::temp_directory_path() / filename;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open fixture path: " + path.string());
    }

    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    out.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));
    if (!binary_blob.empty()) {
        out.write(binary_blob.data(), static_cast<std::streamsize>(binary_blob.size()));
    }
    out.close();
    return path;
}

easy_llm::Config make_test_config() {
    easy_llm::Config config;
    config.num_layers = 1;
    config.hidden_size = 4;
    config.num_heads = 2;
    config.num_heads_kv = 1;
    config.vocab_size = 10;
    config.architecture = "Qwen2ForCausalLM";
    config.model_type = "qwen2";
    return config;
}

std::vector<TensorSpec> make_complete_specs() {
    return {
        {"model.embed_tokens.weight", {10, 4}, 1.0f},
        {"model.layers.0.self_attn.q_proj.weight", {4, 4}, 2.0f},
        {"model.layers.0.self_attn.k_proj.weight", {2, 4}, 3.0f},
        {"model.layers.0.self_attn.v_proj.weight", {2, 4}, 4.0f},
        {"model.layers.0.self_attn.o_proj.weight", {4, 4}, 5.0f},
        {"model.layers.0.input_layernorm.weight", {4}, 6.0f},
        {"model.layers.0.mlp.down_proj.weight", {4, 8}, 7.0f},
        {"model.layers.0.mlp.gate_proj.weight", {8, 4}, 8.0f},
        {"model.layers.0.mlp.up_proj.weight", {8, 4}, 9.0f},
        {"model.layers.0.post_attention_layernorm.weight", {4}, 10.0f},
        {"model.norm.weight", {4}, 11.0f}
    };
}

void consume_required_weights(const easy_llm::Config& config,
                              const easy_llm::LayerKeyPrefix& key_prefix,
                              easy_llm::ModelParam& model_param) {
    (void)model_param.take_param("model.embed_tokens.weight");
    for (int i = 0; i < config.num_layers; ++i) {
        const std::string layer_key = key_prefix.layer(i);
        (void)model_param.take_param(key_prefix.self_attn_q_proj(layer_key) + ".weight");
        (void)model_param.take_param(key_prefix.self_attn_k_proj(layer_key) + ".weight");
        (void)model_param.take_param(key_prefix.self_attn_v_proj(layer_key) + ".weight");
        (void)model_param.take_param(key_prefix.self_attn_o_proj(layer_key) + ".weight");
        (void)model_param.take_param(key_prefix.input_layer_norm(layer_key) + ".weight");
        (void)model_param.take_param(key_prefix.mlp_down_proj(layer_key) + ".weight");
        (void)model_param.take_param(key_prefix.mlp_gate_proj(layer_key) + ".weight");
        (void)model_param.take_param(key_prefix.mlp_up_proj(layer_key) + ".weight");
        (void)model_param.take_param(key_prefix.post_attention_layer_norm(layer_key) + ".weight");
    }
    (void)model_param.take_param(key_prefix.model_norm() + ".weight");
}

template <typename ExceptionType, typename Fn>
bool expect_throws(Fn&& fn, const char* case_name) {
    try {
        fn();
    } catch (const ExceptionType&) {
        return true;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << case_name << " threw wrong exception: " << e.what() << "\n";
        return false;
    }
    std::cerr << "FAIL: " << case_name << " should throw\n";
    return false;
}

bool run_take_and_peek_contract_case() {
    auto specs = make_complete_specs();
    std::filesystem::path path = write_safetensors_fixture("easy_llm_model_param_contract.safetensors", specs);
    auto model_param = easy_llm::ModelParam::load(path.string());

    const std::string key = "model.embed_tokens.weight";
    if (!model_param->contains(key)) {
        std::cerr << "FAIL: contains should be true before take\n";
        return false;
    }
    const auto& peeked = model_param->peek_param(key);
    if (peeked.shape() != std::vector<int>({10, 4})) {
        std::cerr << "FAIL: peek shape mismatch\n";
        return false;
    }
    auto taken = model_param->take_param(key);
    if (taken.shape() != std::vector<int>({10, 4})) {
        std::cerr << "FAIL: take shape mismatch\n";
        return false;
    }
    if (model_param->contains(key)) {
        std::cerr << "FAIL: contains should be false after take\n";
        return false;
    }
    if (!expect_throws<std::out_of_range>([&]() {
            (void)model_param->peek_param(key);
        }, "peek after take")) {
        return false;
    }
    return true;
}

bool run_validation_success_case() {
    auto specs = make_complete_specs();
    std::filesystem::path path = write_safetensors_fixture("easy_llm_model_param_valid.safetensors", specs);
    auto model_param = easy_llm::ModelParam::load(path.string());
    auto config = make_test_config();
    auto key_prefix = easy_llm::create_layer_key_prefix(config);
    easy_llm::validate_model_params_before_load(config, *key_prefix, *model_param);
    return true;
}

bool run_validation_missing_key_case() {
    auto specs = make_complete_specs();
    specs.erase(specs.begin() + 2);  // remove k_proj
    std::filesystem::path path = write_safetensors_fixture("easy_llm_model_param_missing.safetensors", specs);
    auto model_param = easy_llm::ModelParam::load(path.string());
    auto config = make_test_config();
    auto key_prefix = easy_llm::create_layer_key_prefix(config);
    return expect_throws<std::invalid_argument>([&]() {
        easy_llm::validate_model_params_before_load(config, *key_prefix, *model_param);
    }, "validation missing key");
}

bool run_validation_bad_shape_case() {
    auto specs = make_complete_specs();
    specs[0].shape = {9, 4};  // embed vocab mismatch
    std::filesystem::path path = write_safetensors_fixture("easy_llm_model_param_bad_shape.safetensors", specs);
    auto model_param = easy_llm::ModelParam::load(path.string());
    auto config = make_test_config();
    auto key_prefix = easy_llm::create_layer_key_prefix(config);
    return expect_throws<std::invalid_argument>([&]() {
        easy_llm::validate_model_params_before_load(config, *key_prefix, *model_param);
    }, "validation bad shape");
}

bool run_remaining_keys_success_case() {
    auto specs = make_complete_specs();
    std::filesystem::path path = write_safetensors_fixture("easy_llm_model_param_remaining_ok.safetensors", specs);
    auto model_param = easy_llm::ModelParam::load(path.string());
    auto config = make_test_config();
    auto key_prefix = easy_llm::create_layer_key_prefix(config);
    easy_llm::validate_model_params_before_load(config, *key_prefix, *model_param);
    consume_required_weights(config, *key_prefix, *model_param);
    easy_llm::validate_no_remaining_model_params(*model_param);
    if (model_param->size() != 0) {
        std::cerr << "FAIL: remaining key count should be 0\n";
        return false;
    }
    return true;
}

bool run_remaining_keys_fail_case() {
    auto specs = make_complete_specs();
    specs.push_back({"model.unused_adapter.weight", {2, 2}, 12.0f});
    std::filesystem::path path = write_safetensors_fixture("easy_llm_model_param_remaining_bad.safetensors", specs);
    auto model_param = easy_llm::ModelParam::load(path.string());
    auto config = make_test_config();
    auto key_prefix = easy_llm::create_layer_key_prefix(config);
    easy_llm::validate_model_params_before_load(config, *key_prefix, *model_param);
    consume_required_weights(config, *key_prefix, *model_param);
    easy_llm::validate_no_remaining_model_params(*model_param);
    if (model_param->size() != 1) {
        std::cerr << "FAIL: remaining key count should be 1\n";
        return false;
    }
    std::vector<std::string> remaining = model_param->remaining_keys();
    if (remaining.size() != 1 || remaining[0] != "model.unused_adapter.weight") {
        std::cerr << "FAIL: remaining key should be model.unused_adapter.weight\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    if (!run_take_and_peek_contract_case()) {
        return 1;
    }
    if (!run_validation_success_case()) {
        return 1;
    }
    if (!run_validation_missing_key_case()) {
        return 1;
    }
    if (!run_validation_bad_shape_case()) {
        return 1;
    }
    if (!run_remaining_keys_success_case()) {
        return 1;
    }
    if (!run_remaining_keys_fail_case()) {
        return 1;
    }
    std::cout << "PASS: model param validation test\n";
    return 0;
}
