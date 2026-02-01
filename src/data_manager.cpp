#include "data_manager.hpp"

#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>

#include "spdlog/spdlog.h"

#include "tokenizer.hpp"
#include "utils.hpp"

namespace easy_gpt {

using std::string;
using std::vector;
using std::unique_ptr;
using std::move;

namespace {

std::string escape_single_line(std::string text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (char ch : text) {
        switch (ch) {
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            default:
                escaped += ch;
                break;
        }
    }
    return escaped;
}

}  // namespace

DataManager::DataManager(unique_ptr<Tokenizer>&& tokenizer) : 
    tokenizer_{move(tokenizer)} {
    pad_id_ = tokenizer_->get_pad_id();
    spdlog::info("DataManager created");
}   

void DataManager::add_input(InputSample&& input) {
    inputs_.emplace_back(move(input));
    outputs_.emplace_back(OutputSample{"", {}, {}, 0});
}

vector<vector<int>> DataManager::get_inputs() {
    vector<vector<int>> batch{};
    tokenize_inputs(batch);
    int max_len = compute_batch_padding(batch);
    spdlog::info("Batch size: {}", batch.size());
    for (vector<int>& token_ids : batch) {
        spdlog::debug("token_id before pad: {}", fmt::join(token_ids, ", "));
    }
    apply_padding(batch, max_len);
    for (vector<int>& token_ids : batch) {
        spdlog::debug("token_id after pad: {}", fmt::join(token_ids, ", "));
    }
    return batch;
}

void DataManager::add_output_token(const std::pair<vector<float>, vector<int>>& softmax_info,
                                   int step,
                                   const std::vector<int>& sample_ids,
                                   const std::vector<int>& sampled_token_ids) {
    const auto& [softmax_probs, softmax_shape] = softmax_info;
    if (softmax_shape.size() < 3) {
        throw std::invalid_argument("Output shape must be [batch, seq, vocab].");
    }
    int batch_size = softmax_shape[0];
    int seq_len = softmax_shape[1];
    int vocab_size = softmax_shape[2];
    if (batch_size != static_cast<int>(sample_ids.size())) {
        throw std::invalid_argument("sample_ids size must match batch size.");
    }
    if (batch_size != static_cast<int>(sampled_token_ids.size())) {
        throw std::invalid_argument("sampled_token_ids size must match batch size.");
    }
    if (seq_len != 1) {
        throw std::invalid_argument("sampled_token_ids expects seq_len == 1.");
    }
    (void)softmax_probs;
    (void)vocab_size;

    for (int b = 0; b < batch_size; ++b) {
        int sample_id = sample_ids[b];
        if (sample_id < 0 || sample_id >= static_cast<int>(inputs_.size())) {
            continue;
        }
        if (!should_sample_step(sample_id, step)) {
            continue;
        }
        int token_id = sampled_token_ids[b];
        std::string token_str = decode_token_raw(token_id);
        std::string token_decoded = tokenizer_->decode({token_id});
        spdlog::info("Sampled token: id = {}, str = {}, decoded = {}", token_id, token_str, token_decoded);
        append_output_token(outputs_[sample_id], token_id, token_str);
    }
}

bool DataManager::should_sample_step(int sample_id, int step) const {
    return step >= static_cast<int>(inputs_[sample_id].token_ids.size()) - 1;
}

std::string DataManager::decode_token_raw(int token_id) const {
    return tokenizer_->ids_to_tokens({token_id})[0];
}

bool DataManager::append_output_token(OutputSample& output, int token_id, const std::string& token_str) const {
    if (token_id == pad_id_) {
        return false;
    }
    output.token_ids.push_back(token_id);
    output.tokens.push_back(token_str);
    output.token_count = static_cast<int>(output.token_ids.size());
    return true;
}

void DataManager::log_outputs(const std::string& output_path) {
    std::ofstream out;
    if (!output_path.empty()) {
        std::filesystem::path path(output_path);
        std::error_code ec;
        if (!path.parent_path().empty()) {
            std::filesystem::create_directories(path.parent_path(), ec);
        }
        out.open(path);
        if (!out.is_open()) {
            spdlog::error("Failed to open output file: {}", output_path);
        }
    }
    for (auto& output : outputs_) {
        // Use tokenizer's decode to properly handle byte mapping (including Ġ -> space)
        output.text = tokenizer_->decode(output.token_ids);
        spdlog::info("Output text: {}", output.text);
        if (out.is_open()) {
            out << escape_single_line(output.text) << "\n";
        }
    }
}

std::vector<int> DataManager::get_seq_lens() const {
    std::vector<int> seq_lens;
    seq_lens.reserve(inputs_.size());
    for (const auto& input : inputs_) {
        seq_lens.push_back(input.seq_len);
    }
    return seq_lens;
}

std::vector<int> DataManager::get_pad_lens() const {
    std::vector<int> pad_lens;
    pad_lens.reserve(inputs_.size());
    for (const auto& input : inputs_) {
        pad_lens.push_back(input.pad_len);
    }
    return pad_lens;
}

void DataManager::preprocess(InputSample& input) {
    input.text = input.original_text;
}

void DataManager::tokenize_inputs(vector<vector<int>>& batch) {
    batch.reserve(inputs_.size());
    for (InputSample& input : inputs_) {
        preprocess(input);
        input.tokens = tokenizer_->tokenize(input.text);
        input.token_ids = tokenizer_->tokens_to_ids(input.tokens);
        batch.emplace_back(input.token_ids);
    }
}

int DataManager::compute_batch_padding(const vector<vector<int>>& batch) {
    int max_len = 0;
    for (const vector<int>& token_ids : batch) {
        max_len = std::max(max_len, static_cast<int>(token_ids.size()));
    }
    for (size_t i = 0; i < inputs_.size(); ++i) {
        int seq_len = static_cast<int>(batch[i].size());
        inputs_[i].seq_len = seq_len;
        inputs_[i].pad_len = max_len - seq_len;
    }
    return max_len;
}

void DataManager::apply_padding(vector<vector<int>>& batch, int max_len) {
    spdlog::info("Padding batch");
    for (vector<int>& token_ids : batch) {
        int pad_len = max_len - static_cast<int>(token_ids.size());
        if (pad_len <= 0) {
            continue;
        }
        token_ids.insert(token_ids.begin(), pad_len, pad_id_);
    }
}

} // namespace easy_gpt
