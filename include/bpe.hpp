#ifndef EASY_GPT_BPE_HPP
#define EASY_GPT_BPE_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace easy_gpt {

class Bpe {
public:
    Bpe(const std::vector<std::string>& token_merges);
    Bpe(const Bpe& other) = default;
    Bpe(Bpe&& other) noexcept = default;
    Bpe& operator=(const Bpe& other) = default;
    Bpe& operator=(Bpe&& other) noexcept = default;
    std::vector<std::string> encode(const std::string& text) const;
    void encode_into(const std::string& text, bool& first_word, bool suppress_prefix_for_first_word,
                     std::vector<std::string>& out) const;
    std::string decode(const std::vector<std::string>& tokens) const;

private:
    void load_merges(const std::string& path);
    void load_merges(const std::vector<std::string>& token_merges);
    std::string apply_bpe(const std::string& word) const;
    void init_byte_encoder();
    void build_merge_ranks();

    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> merge_ranks_;
    std::unordered_map<unsigned char, std::string> byte_encoder_;
    std::unordered_map<std::string, unsigned char> byte_decoder_;
    mutable std::unordered_map<std::string, std::string> bpe_cache_;
};
} // namespace easy_gpt

#endif // EASY_GPT_BPE_HPP
