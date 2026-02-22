#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "config.hpp"
#include "data_manager.hpp"
#include "tokenizer.hpp"

namespace {

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to write test file: " + path.string());
    }
    out << content;
}

std::vector<std::string> read_lines(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("failed to read test file: " + path.string());
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    return lines;
}

std::unique_ptr<easy_llm::Tokenizer> create_tokenizer(const std::filesystem::path& tokenizer_path,
                                                      const std::filesystem::path& tokenizer_config_path) {
    easy_llm::Config config;
    config.bos_token_id = 5;
    config.tokenizer_path = tokenizer_path.string();
    config.tokenizer_config_path = tokenizer_config_path.string();
    return easy_llm::Tokenizer::create(config);
}

bool expect_eq(const std::vector<int>& actual,
               const std::vector<int>& expected,
               const char* case_name) {
    if (actual != expected) {
        std::cerr << "FAIL: " << case_name << " mismatch\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    using namespace easy_llm;

    const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "easy_llm_data_manager_invariants";
    std::filesystem::create_directories(tmp);

    const std::filesystem::path tokenizer_path = tmp / "tokenizer.json";
    const std::filesystem::path tokenizer_config_path = tmp / "tokenizer_config.json";
    const std::filesystem::path output_path = tmp / "outputs.txt";

    write_file(tokenizer_path, R"json(
{
  "model": {
    "vocab": {
      "<pad>": 0,
      "a": 1,
      "b": 2,
      "c": 3,
      "d": 4
    },
    "merges": []
  }
}
)json");

    write_file(tokenizer_config_path, R"json(
{
  "pad_token": "<pad>",
  "added_tokens_decoder": {
    "0": { "content": "<pad>", "special": true }
  }
}
)json");

    DataManager dm(create_tokenizer(tokenizer_path, tokenizer_config_path));
    dm.add_input(InputSample("a"));
    dm.add_input(InputSample("ab"));

    std::vector<std::vector<int>> batch = dm.get_inputs();
    if (batch.size() != 2) {
        std::cerr << "FAIL: batch size mismatch\n";
        return 1;
    }
    if (!expect_eq(batch[0], {0, 5, 1}, "left padding sample 0")) {
        return 1;
    }
    if (!expect_eq(batch[1], {5, 1, 2}, "left padding sample 1")) {
        return 1;
    }
    if (!expect_eq(dm.get_seq_lens(), {2, 3}, "seq lens")) {
        return 1;
    }
    if (!expect_eq(dm.get_pad_lens(), {1, 0}, "pad lens")) {
        return 1;
    }

    const std::pair<std::vector<float>, std::vector<int>> softmax_info{
        std::vector<float>(2 * 1 * 5, 0.0f),
        std::vector<int>{2, 1, 5}
    };
    dm.add_output_token(softmax_info, 1, {0, 1}, {1, 2});
    dm.add_output_token(softmax_info, 2, {0, 1}, {3, 4});
    dm.add_output_token(softmax_info, 3, {0, 1}, {0, 0});

    dm.log_outputs(output_path.string());
    const auto lines = read_lines(output_path);
    if (lines.size() != 2) {
        std::cerr << "FAIL: output sample size mismatch\n";
        return 1;
    }
    if (lines[0] != "ac") {
        std::cerr << "FAIL: sample 0 generated text mismatch, got=" << lines[0] << "\n";
        return 1;
    }
    if (lines[1] != "d") {
        std::cerr << "FAIL: sample 1 generated text mismatch, got=" << lines[1] << "\n";
        return 1;
    }

    std::cout << "PASS: data manager invariants test\n";
    return 0;
}
