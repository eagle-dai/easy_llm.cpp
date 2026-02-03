#ifndef EASY_LLM_UTILS_HPP
#define EASY_LLM_UTILS_HPP

#include <string>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <random>

#include "json.hpp"

namespace easy_llm {

extern std::mt19937 gen;
extern std::uniform_real_distribution<> dis;

namespace utils {

std::string strip(const std::string& text);
std::string replace(const std::string& text, const std::string& old_str, const std::string& new_str);
std::vector<std::string> split(const std::string& text, const std::string& delim);
std::vector<std::string> sent2chars(const std::string& text);
std::string unicode_to_utf8(int codepoint);
nlohmann::json load_json(const std::string& json_path);
std::vector<char> read_bytes(std::ifstream& file, size_t num_bytes);

}

} // namespace easy_llm

#endif // EASY_LLM_UTILS_HPP
