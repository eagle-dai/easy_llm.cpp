#include "bpe.hpp"

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>

#include "utils.hpp"
#include "spdlog/spdlog.h"

namespace easy_gpt {

using std::string;
using std::vector;

Bpe::Bpe(const vector<string>& token_merges) {
    init_byte_encoder();
    load_merges(token_merges);
}

vector<string> Bpe::encode(const string& text) const {
    vector<string> tokens{};
    bool first_word = true;
    encode_into(text, first_word, false, tokens);
    return tokens;
}

void Bpe::encode_into(const string& text, bool& first_word, bool suppress_prefix_for_first_word,
                      vector<string>& out) const {
    std::istringstream iss(text);
    string word;
    bool local_first = first_word || suppress_prefix_for_first_word;
    bool saw_word = false;
    while (iss >> word) {
        string raw_word = word;
        if (!local_first) {
            raw_word = " " + raw_word;
        }
        local_first = false;
        saw_word = true;

        string mapped_word = "";
        for (char c : raw_word) {
            auto uc = static_cast<unsigned char>(c);
            auto it = byte_encoder_.find(uc);
            if (it != byte_encoder_.end()) {
                mapped_word += it->second;
            }
        }

        for (const auto& token: utils::split(apply_bpe(mapped_word), " ")) {
            out.push_back(token);
        }
    }
    if (saw_word) {
        first_word = false;
    }
}

string Bpe::decode(const vector<string>& tokens) const {
    string text = "";
    for (const auto& token : tokens) {
        text += token;
    }
    string decoded_text = "";
    auto chars = utils::sent2chars(text);
    for (const auto& c : chars) {
        auto it = byte_decoder_.find(c);
        if (it != byte_decoder_.end()) {
            decoded_text += static_cast<char>(it->second);
        } else {
            decoded_text += c;
        }
    }
    return decoded_text;
}

void Bpe::load_merges(const string& path) {
    std::ifstream file(path);
    string line;
    while (std::getline(file, line)) {
        string stripped = utils::strip(line);
        if (stripped.empty() || stripped[0] == '#') {
            continue;
        }
        std::istringstream iss(stripped);
        string left, right;
        if (iss >> left >> right) {
            merges_.emplace_back(left, right);
        }
    }
    build_merge_ranks();
}

void Bpe::load_merges(const vector<string>& token_merges) {
    for (const string& merge : token_merges) {
        string left, right;
        std::istringstream iss(merge);
        iss >> left >> right;
        merges_.emplace_back(left, right);
    }
    build_merge_ranks();
}

void Bpe::build_merge_ranks() {
    merge_ranks_.clear();
    merge_ranks_.reserve(merges_.size());
    for (int i = 0; i < static_cast<int>(merges_.size()); ++i) {
        const auto& left_token = merges_[i].first;
        const auto& right_token = merges_[i].second;
        merge_ranks_[left_token].emplace(right_token, i);
    }
}

std::string Bpe::apply_bpe(const std::string& word) const {
    if (auto it = bpe_cache_.find(word); it != bpe_cache_.end()) {
        return it->second;
    }
    auto chars = utils::sent2chars(word);
    if (chars.empty()) return "";

    while (true) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_i = static_cast<size_t>(-1);

        for (size_t j = 0; j + 1 < chars.size(); ++j) {
            auto it1 = merge_ranks_.find(chars[j]);
            if (it1 == merge_ranks_.end()) continue;
            auto it2 = it1->second.find(chars[j + 1]);
            if (it2 == it1->second.end()) continue;
            const int rank = it2->second;
            if (rank < best_rank) {
                best_rank = rank;
                best_i = j;
                // can’t break: still need to find a smaller rank
            }
        }
        if (best_i == static_cast<size_t>(-1)) break;  // no mergeable pairs

        chars[best_i] += chars[best_i + 1];               // 少一次临时对象
        chars.erase(chars.begin() + best_i + 1);          // 低改动版本仍然 erase
    }

    // reserve in advance to avoid repeated scaling.
    size_t total = 0;
    for (const auto& s : chars) total += s.size();
    if (chars.size() > 1) total += (chars.size() - 1);  // spaces
    std::string out;
    out.reserve(total);
    for (size_t i = 0; i < chars.size(); ++i) {
        if (i) out.push_back(' ');
        out += chars[i];
    }

    bpe_cache_.emplace(word, out);
    return out;
}

void Bpe::init_byte_encoder() {
    vector<int> byte_values;
    byte_values.reserve(256);
    // byte_values = [ord("!"), ord("~")+1) + [ord("¡"), ord("¬")+1) + [ord("®"), ord("ÿ")+1)
    for (int i = '!'; i <= '~'; ++i) byte_values.push_back(i);
    for (int i = 0xA1; i <= 0xAC; ++i) byte_values.push_back(i);
    for (int i = 0xAE; i <= 0xFF; ++i) byte_values.push_back(i);

    auto unicode_values = byte_values;
    unicode_values.reserve(256);
    int added_count = 0;
    for (int byte_value = 0; byte_value < 256; ++byte_value) {
        bool found = false;
        for (int val : byte_values) {
            if (val == byte_value) {
                found = true;
                break;
            }
        }
        if (!found) {
            byte_values.push_back(byte_value);
            unicode_values.push_back(256 + added_count);
            added_count++;
        }
    }

    for (size_t i = 0; i < byte_values.size(); ++i) {
        auto byte_value = static_cast<unsigned char>(byte_values[i]);
        auto unicode_str = utils::unicode_to_utf8(unicode_values[i]);
        byte_encoder_[byte_value] = unicode_str;
        byte_decoder_[unicode_str] = byte_value;
    }
}

} // namespace easy_gpt
