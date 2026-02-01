#include "models/loader.hpp"

#include <random>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

#include "spdlog/spdlog.h"

#include "utils.hpp"
#include "ops.hpp"
#include "tensor.hpp"

namespace easy_gpt {

using std::vector;
using std::string;
using std::array;
using std::move;

namespace {

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

HeaderInfo parse_header(const char* addr, size_t file_size) {
    if (file_size < sizeof(uint64_t)) {
        throw std::runtime_error("File too small to contain header length");
    }
    auto header_length = *reinterpret_cast<const uint64_t*>(addr);
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
    auto shape = value["shape"];
    int height = shape[0].get<int>();
    int width = 1;
    if (shape.size() == 2) {
        width = shape[1].get<int>();
    } else if (shape.size() != 1) {
        throw std::runtime_error("Invalid shape in ModelParam::load_from_ckpt");
    }
    auto num_val = height * width;
    vector<data_type> weights(num_val);
    auto data_offsets = value["data_offsets"];
    auto start_offset = data_offsets[0].get<long long>();
    auto offset = data_base_offset + static_cast<size_t>(start_offset);

    if (offset + num_val * sizeof(data_type) > file_size) {
        throw std::runtime_error("Data offset exceeds file size for key: " + key);
    }
    std::memcpy(weights.data(), addr + offset, num_val * sizeof(data_type));
    return Tensor{weights, {height, width}};
}

void finalize_log(const std::unordered_map<string, Tensor>& params) {
    for (const auto& [key, value] : params) {
        if (key == "model.embed_tokens.weight") {
            spdlog::debug("key: {}, value[:10] == {}", key,
                fmt::join(vector<float>(value.data().begin(), value.data().begin() + 10), ", "));
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

Tensor ModelParam::get_param(const string& key) {
    if (!contains(key)) {
        spdlog::error("Key not found in ModelParam::get_param: {}", key);
        throw std::out_of_range("Key not found in ModelParam::get_param");
    }
    return move(params_[key]);
}

bool ModelParam::contains(const string& key) const {
    auto it = params_.find(key);
    if (it != params_.end()) {
        return true;
    }
    return false;
}

}  // namespace easy_gpt
