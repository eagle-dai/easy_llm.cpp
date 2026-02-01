#include "utils.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <random>

#include "spdlog/spdlog.h"

namespace easy_gpt {

std::mt19937 gen(std::random_device{}());
std::uniform_real_distribution<> dis(-0.1, 0.1);

namespace utils {

using std::string;
using std::vector;
using std::unordered_map;
using std::unordered_set;
using nlohmann::json;

string strip(const string& text) {
    auto start = text.find_first_not_of(" \t\n");
    auto end = text.find_last_not_of(" \t\n");
    if (start == string::npos) return "";
    return text.substr(start, end - start + 1);
}

string replace(const string& text, const string& old_str, const string& new_str) {
    string result{text};
    size_t start{0};
    while ((start = result.find(old_str, start)) != string::npos) {
        result.replace(start, old_str.length(), new_str);
        start += new_str.length();
    }
    return result;
}

vector<string> split(const string& text, const string& delim) {
    if (delim.empty()) {
        throw std::invalid_argument("delimiter must not be empty");
    }
    vector<string> result;
    size_t start = 0;
    while (true) {
        auto pos = text.find(delim, start);
        if (pos == string::npos) {
            if (start < text.length())
                result.emplace_back(text.substr(start));
            else if (start == text.length() && !result.empty() && text.substr(text.length()-delim.length()) == delim) {
            }
            break;
        }
        result.emplace_back(text.substr(start, pos - start));
        start = pos + delim.length();
    }
    return result;
}

vector<string> sent2chars(const string& text) {
    vector<string> chars;
    chars.reserve(text.size());
    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = text[i];
        string char_utf8;
        if (c < 0x80) {
            char_utf8 = text.substr(i, 1);
            i += 1;
        } else if ((c >> 5) == 0x6) { // 2-byte sequence
            char_utf8 = text.substr(i, 2);
            i += 2;
        } else if ((c >> 4) == 0xE) { // 3-byte sequence
            char_utf8 = text.substr(i, 3);
            i += 3;
        } else if ((c >> 3) == 0x1E) { // 4-byte sequence
            char_utf8 = text.substr(i, 4);
            i += 4;
        } else {
            // Simple fallback: treat as single byte to avoid infinite loops
            char_utf8 = text.substr(i, 1);
            i += 1;
        }
        chars.emplace_back(char_utf8);
    }
    return chars;
}

string unicode_to_utf8(int codepoint) {
    string out;
    if (codepoint <= 0x7f)
        out.append(1, static_cast<char>(codepoint));
    else if (codepoint <= 0x7ff) {
        out.append(1, static_cast<char>(0xc0 | ((codepoint >> 6) & 0x1f)));
        out.append(1, static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        out.append(1, static_cast<char>(0xe0 | ((codepoint >> 12) & 0x0f)));
        out.append(1, static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        out.append(1, static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        out.append(1, static_cast<char>(0xf0 | ((codepoint >> 18) & 0x07)));
        out.append(1, static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        out.append(1, static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        out.append(1, static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
    return out;
}

size_t count_chars(const string& str) {
    size_t count{0};
    for (size_t i = 0; i < str.length();) {
        unsigned char c = str[i];
        if ((c & 0x80) == 0) {        // ASCII character
            i += 1;
        } else if ((c & 0xE0) == 0xC0) { // 2-byte UTF-8 sequence
            i += 2;
        } else if ((c & 0xF0) == 0xE0) { // 3-byte UTF-8 sequence (most Chinese characters appear here)
            i += 3;
        } else if ((c & 0xF8) == 0xF0) { // 4-byte UTF-8 sequence
            i += 4;
        } else {                      // Invalid UTF-8 sequence
            i += 1;
        }
        count++;
    }
    return count;
}

json load_json(const string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open()) {
        spdlog::error("Failed to open file: {}", json_path);
        return json{};
    } else {
        auto json_data = json::parse(f);
        return json_data;
    }
}

vector<char> read_bytes(std::ifstream& file, size_t num_bytes) {
    vector<char> buffer(num_bytes);
    file.read(buffer.data(), num_bytes);
    if (file.gcount() != static_cast<std::streamsize>(num_bytes)) {
        throw std::runtime_error("Failed to read the required number of bytes.");
    }
    return buffer;
}

} // namespace utils
} // namespace easy_gpt
