#include "models/loader.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <utility>

#include "spdlog/spdlog.h"

#include "json.hpp"
#include "tensor.hpp"

namespace easy_llm {

using std::vector;
using std::string;

namespace {

enum class SourceDType {
    Bf16,
    F16,
    F32
};

struct MMapGuard {
    const char* addr;
    size_t size;
    int fd;
    ~MMapGuard() {
        if (addr != MAP_FAILED) {
            munmap(const_cast<char*>(addr), size);
        }
        if (fd != -1) {
            close(fd);
        }
    }
};

MMapGuard open_mmap(const string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("Failed to stat file: " + path);
    }
    auto file_size = static_cast<size_t>(sb.st_size);
    const char* addr = static_cast<const char*>(mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (addr == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Failed to mmap file: " + path);
    }
    return MMapGuard{addr, file_size, fd};
}

struct HeaderInfo {
    nlohmann::json header_json;
    size_t data_base_offset;
};

size_t dtype_byte_width(SourceDType dtype) {
    switch (dtype) {
        case SourceDType::Bf16: return sizeof(std::uint16_t);
        case SourceDType::F16: return sizeof(std::uint16_t);
        case SourceDType::F32: return sizeof(float);
        default: throw std::runtime_error("Unknown source dtype.");
    }
}

SourceDType parse_dtype(const string& key, const nlohmann::json& value) {
    if (!value.contains("dtype") || !value["dtype"].is_string()) {
        throw std::runtime_error("Tensor '" + key + "' missing string dtype.");
    }
    const string dtype = value["dtype"].get<string>();
    if (dtype == "BF16") {
        return SourceDType::Bf16;
    }
    if (dtype == "F16") {
        return SourceDType::F16;
    }
    if (dtype == "F32") {
        return SourceDType::F32;
    }
    throw std::runtime_error("Unsupported dtype '" + dtype + "' for tensor '" + key + "'.");
}

std::pair<std::vector<int>, size_t> parse_shape(const string& key, const nlohmann::json& value) {
    if (!value.contains("shape") || !value["shape"].is_array()) {
        throw std::runtime_error("Tensor '" + key + "' missing shape array.");
    }
    const auto& shape_json = value["shape"];
    if (shape_json.empty()) {
        throw std::runtime_error("Tensor '" + key + "' has empty shape; scalar tensors are unsupported.");
    }

    std::vector<int> shape;
    shape.reserve(shape_json.size());
    size_t num_elements = 1;
    for (const auto& dim_json : shape_json) {
        if (!dim_json.is_number_integer()) {
            throw std::runtime_error("Tensor '" + key + "' has non-integer shape dimension.");
        }
        long long dim_ll = dim_json.get<long long>();
        if (dim_ll <= 0 || dim_ll > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Tensor '" + key + "' has invalid shape dimension.");
        }
        int dim = static_cast<int>(dim_ll);
        if (num_elements > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
            throw std::runtime_error("Tensor '" + key + "' shape product overflow.");
        }
        num_elements *= static_cast<size_t>(dim);
        shape.push_back(dim);
    }
    return {shape, num_elements};
}

std::pair<size_t, size_t> parse_data_offsets(const string& key, const nlohmann::json& value) {
    if (!value.contains("data_offsets") || !value["data_offsets"].is_array() || value["data_offsets"].size() != 2) {
        throw std::runtime_error("Tensor '" + key + "' has invalid data_offsets.");
    }
    const auto& offsets = value["data_offsets"];
    if (!offsets[0].is_number_integer() || !offsets[1].is_number_integer()) {
        throw std::runtime_error("Tensor '" + key + "' has non-integer data_offsets.");
    }
    long long begin_ll = offsets[0].get<long long>();
    long long end_ll = offsets[1].get<long long>();
    if (begin_ll < 0 || end_ll < begin_ll) {
        throw std::runtime_error("Tensor '" + key + "' has invalid data_offsets range.");
    }
    return {static_cast<size_t>(begin_ll), static_cast<size_t>(end_ll)};
}

float fp16_to_float(std::uint16_t fp16) {
    const std::uint32_t sign = static_cast<std::uint32_t>(fp16 & 0x8000u) << 16;
    std::uint32_t exp = (fp16 >> 10) & 0x1Fu;
    std::uint32_t mant = fp16 & 0x03FFu;
    std::uint32_t fp32_bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            fp32_bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03FFu;
            const std::uint32_t exp32 = exp + (127 - 15);
            fp32_bits = sign | (exp32 << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        fp32_bits = sign | 0x7F800000u | (mant << 13);
    } else {
        const std::uint32_t exp32 = exp + (127 - 15);
        fp32_bits = sign | (exp32 << 23) | (mant << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &fp32_bits, sizeof(result));
    return result;
}

vector<data_type> decode_tensor_data(const string& key, const char* addr, size_t abs_begin_offset,
                                     size_t num_elements, SourceDType dtype) {
    vector<data_type> weights(num_elements);
    const size_t elem_bytes = dtype_byte_width(dtype);
    const char* src = addr + abs_begin_offset;

    for (size_t i = 0; i < num_elements; ++i) {
        const char* elem_ptr = src + i * elem_bytes;
        float val = 0.0f;
        if (dtype == SourceDType::Bf16) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, elem_ptr, sizeof(bits));
            val = bf16_to_float(bits);
        } else if (dtype == SourceDType::F16) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, elem_ptr, sizeof(bits));
            val = fp16_to_float(bits);
        } else if (dtype == SourceDType::F32) {
            std::memcpy(&val, elem_ptr, sizeof(val));
        } else {
            throw std::runtime_error("Unknown source dtype for tensor '" + key + "'.");
        }
        weights[i] = data_type(val);
    }
    return weights;
}

bool source_matches_target_dtype(SourceDType dtype) {
#ifdef USE_BF16
    return dtype == SourceDType::Bf16;
#elif defined(USE_FP16)
    return dtype == SourceDType::F16;
#elif defined(USE_FP32)
    return dtype == SourceDType::F32;
#else
    (void)dtype;
    return false;
#endif
}

HeaderInfo parse_header(const char* addr, size_t file_size) {
    if (file_size < sizeof(uint64_t)) {
        throw std::runtime_error("File too small to contain header length");
    }
    uint64_t header_length = 0;
    std::memcpy(&header_length, addr, sizeof(header_length));
    if (sizeof(uint64_t) + header_length > file_size) {
        throw std::runtime_error("Header length exceeds file size");
    }
    string header_str(addr + sizeof(uint64_t), header_length);
    auto header_json = nlohmann::json::parse(header_str);
    auto data_base_offset = sizeof(uint64_t) + header_length;

    return HeaderInfo{std::move(header_json), data_base_offset};
}

Tensor load_tensor(const string& key, const nlohmann::json& value, const char* addr,
    size_t file_size, size_t data_base_offset) {
    const SourceDType dtype = parse_dtype(key, value);
    const auto [shape, num_elements] = parse_shape(key, value);
    const auto [data_begin, data_end] = parse_data_offsets(key, value);
    if (data_begin > data_end) {
        throw std::runtime_error("Tensor '" + key + "' has invalid data_offsets range.");
    }

    if (data_base_offset > file_size || data_end > file_size - data_base_offset) {
        throw std::runtime_error("Data offset exceeds file size for key: " + key);
    }

    const size_t elem_bytes = dtype_byte_width(dtype);
    if (num_elements > std::numeric_limits<size_t>::max() / elem_bytes) {
        throw std::runtime_error("Tensor '" + key + "' byte size overflow.");
    }
    const size_t expected_bytes = num_elements * elem_bytes;
    const size_t actual_bytes = data_end - data_begin;
    if (actual_bytes != expected_bytes) {
        throw std::runtime_error(
            "Tensor '" + key + "' data byte size mismatch: expected " + std::to_string(expected_bytes) +
            ", got " + std::to_string(actual_bytes));
    }

    const size_t abs_begin_offset = data_base_offset + data_begin;
    vector<data_type> weights;
    if (source_matches_target_dtype(dtype)) {
        if (num_elements > std::numeric_limits<size_t>::max() / sizeof(data_type)) {
            throw std::runtime_error("Tensor '" + key + "' target byte size overflow.");
        }
        const size_t target_bytes = num_elements * sizeof(data_type);
        if (target_bytes != actual_bytes) {
            throw std::runtime_error("Tensor '" + key + "' source/target byte mismatch.");
        }
        weights.resize(num_elements);
        std::memcpy(weights.data(), addr + abs_begin_offset, target_bytes);
    } else {
        weights = decode_tensor_data(key, addr, abs_begin_offset, num_elements, dtype);
    }
    return Tensor{weights, shape};
}

void finalize_log(const std::unordered_map<string, Tensor>& params) {
    for (const auto& [key, value] : params) {
        if (key == "model.embed_tokens.weight") {
            const size_t preview = std::min<size_t>(10, static_cast<size_t>(value.size()));
            std::vector<float> sample;
            sample.reserve(preview);
            for (size_t i = 0; i < preview; ++i) {
                sample.push_back(static_cast<float>(value.data()[i]));
            }
            spdlog::debug("key: {}, value[:10] == {}", key,
                fmt::join(sample, ", "));
        }
    }
}

}  // namespace

ModelParam::ModelParam() = default;

std::unique_ptr<ModelParam> ModelParam::load(const string& model_path) {
    auto model_param = std::make_unique<ModelParam>();
    model_param->load_from_ckpt(model_path);
    spdlog::info("ModelParam created");
    return model_param;
}

void ModelParam::load_from_ckpt(const string& path) {
    auto mmap_guard = open_mmap(path);
    auto header_info = parse_header(mmap_guard.addr, mmap_guard.size);
    for (auto& [key, value] : header_info.header_json.items()) {
        if (key == "__metadata__") continue;
        /* Each value is a json containing Tensor details: [data_offsets, dtype, shape],
        where data_offsets is an array of the byte offsets within the ckpt file,
        and for HuggingFace's Qwen2.5 0.5B the dtype is BF16; */
        params_.emplace(key,
            load_tensor(key, value, mmap_guard.addr, mmap_guard.size, header_info.data_base_offset));
    }
    finalize_log(params_);
}

Tensor ModelParam::take_param(const string& key) {
    auto it = params_.find(key);
    if (it == params_.end()) {
        spdlog::error("Key not found in ModelParam::take_param: {}", key);
        throw std::out_of_range("Key not found in ModelParam::take_param: " + key);
    }
    Tensor value = std::move(it->second);
    params_.erase(it);
    return value;
}

const Tensor& ModelParam::peek_param(const string& key) const {
    auto it = params_.find(key);
    if (it == params_.end()) {
        spdlog::error("Key not found in ModelParam::peek_param: {}", key);
        throw std::out_of_range("Key not found in ModelParam::peek_param: " + key);
    }
    return it->second;
}

bool ModelParam::contains(const string& key) const {
    auto it = params_.find(key);
    if (it != params_.end()) {
        return true;
    }
    return false;
}

std::vector<std::string> ModelParam::remaining_keys() const {
    std::vector<std::string> keys;
    keys.reserve(params_.size());
    for (const auto& [key, _] : params_) {
        (void)_;
        keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

std::size_t ModelParam::size() const {
    return params_.size();
}

}  // namespace easy_llm
