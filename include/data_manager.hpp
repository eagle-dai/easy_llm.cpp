#ifndef EASY_GPT_DATA_MANAGER_HPP
#define EASY_GPT_DATA_MANAGER_HPP

#include <string>
#include <vector>
#include <memory>

#include "tokenizer.hpp"
#include "tensor.hpp"

namespace easy_gpt {

struct InputSample {
    InputSample(std::string original_text) : 
        original_text(original_text) {}
    std::string original_text;
    std::string text;
    std::vector<std::string> tokens;
    std::vector<int> token_ids;
    int token_count;
    int seq_len{0};
    int pad_len{0};
    
};

struct OutputSample {
    std::string text;
    std::vector<std::string> tokens;
    std::vector<int> token_ids;
    int token_count;
    
};

class DataManager {
public:
    DataManager(std::unique_ptr<Tokenizer>&& tokenizer);
    DataManager(DataManager&& other) noexcept = default;
    DataManager& operator=(DataManager&& other) noexcept = default;
    void add_input(InputSample&& input);
    void add_output_token(const std::pair<std::vector<float>, std::vector<int>>& softmax_info,
                          int step,
                          const std::vector<int>& sample_ids,
                          const std::vector<int>& sampled_token_ids);
    std::vector<std::vector<int>> get_inputs();
    std::vector<int> get_seq_lens() const;
    std::vector<int> get_pad_lens() const;
    void log_outputs(const std::string& output_path = "");

private:
    std::vector<InputSample> inputs_;
    std::vector<OutputSample> outputs_;
    std::unique_ptr<Tokenizer> tokenizer_;
    int pad_id_{-1};

    void preprocess(InputSample& input);
    void tokenize_inputs(std::vector<std::vector<int>>& batch);
    int compute_batch_padding(const std::vector<std::vector<int>>& batch);
    void apply_padding(std::vector<std::vector<int>>& batch, int max_len);
    bool should_sample_step(int sample_id, int step) const;
    std::string decode_token_raw(int token_id) const;
    bool append_output_token(OutputSample& output, int token_id, const std::string& token_str) const;
};

} // namespace easy_gpt

#endif // EASY_GPT_DATA_MANAGER_HPP
