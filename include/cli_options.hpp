#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace easy_gpt {

struct CliOptions {
    std::string prompt_file;
    std::string prompt;
    int max_steps{100};
    float temperature{0.8f};
    float top_p{0.95f};
    int top_k{20};
    int seed{42};
    bool use_greedy{false};
    bool show_help{false};
};

void print_usage(std::ostream& os);
bool parse_args(int argc, char** argv, CliOptions* options, std::string* error);
bool read_prompts_file(const std::string& path, std::vector<std::string>* prompts, std::string* error);
std::string apply_chat_template(const std::string& user_query);

}  // namespace easy_gpt
