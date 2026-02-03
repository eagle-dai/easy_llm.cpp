#ifndef EASY_LLM_TOKENIZER_HPP
#define EASY_LLM_TOKENIZER_HPP

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "json.hpp"
#include "bpe.hpp"

namespace easy_llm {

struct Config;

class Tokenizer {
public:
    static std::unique_ptr<Tokenizer> create(const Config& config);
    Tokenizer() = default;
    Tokenizer(Tokenizer&& other) noexcept = default;
    Tokenizer& operator=(Tokenizer&& other) noexcept = default;
    std::vector<std::string> tokenize(const std::string& text) const;
    std::vector<int> tokens_to_ids(const std::vector<std::string>& tokens) const;
    std::vector<std::string> ids_to_tokens(const std::vector<int>& ids) const;
    std::string decode(const std::vector<int>& ids) const;
    int get_pad_id() const { return pad_id_; }

private:
    void init_from_config(const Config& config);
    void init_tokenizer_config(const std::string& tokenizer_path);
    void init_special_tokens(const std::string& tokenizer_config_path);
    struct AddedTokenInfo {
        int id;
        std::string token;
        bool is_special;
    };
    std::vector<AddedTokenInfo> parse_added_tokens(const nlohmann::json& tokenizer_config) const;
    void register_added_tokens(const std::vector<AddedTokenInfo>& added_tokens);
    bool parse_pad_id(const nlohmann::json& tokenizer_config);
    void sort_special_tokens();

    std::size_t max_seq_len_{512};
    nlohmann::json tokenizer_config_;
    std::vector<std::string> token_merges_;
    std::unique_ptr<Bpe> bpe_;
    std::unordered_map<std::string, int> token2id_;
    std::unordered_map<int, std::string> id2token_;
    std::vector<std::string> special_tokens_;
    int pad_id_{-1};
    int bos_token_id_{-1};
};

} // namespace easy_llm

#endif // EASY_LLM_TOKENIZER_HPP
