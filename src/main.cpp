#include "spdlog/spdlog.h"

#include <iostream>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "cli_options.hpp"
#include "gpt_engine.hpp"
#include "config.hpp"
#include "data_manager.hpp"
#include "tokenizer.hpp"
#include "models/gpt_model.hpp"
#include "models/loader.hpp"

using std::make_unique;
using std::move;
using std::string;
using std::vector;

int main(int argc, char** argv) {
    spdlog::set_level(spdlog::level::debug);
    spdlog::info("start gpt_engine running");

    easy_gpt::CliOptions options;
    string parse_error;
    if (!easy_gpt::parse_args(argc, argv, &options, &parse_error)) {
        spdlog::error(parse_error);
        easy_gpt::print_usage(std::cerr);
        return 1;
    }
    if (options.show_help) {
        easy_gpt::print_usage(std::cout);
        return 0;
    }

    vector<string> prompts;
    if (!options.prompt_file.empty()) {
        string file_error;
        if (!easy_gpt::read_prompts_file(options.prompt_file, &prompts, &file_error)) {
            spdlog::error(file_error);
            return 1;
        }
    }
    if (!options.prompt.empty()) {
        prompts.push_back(move(options.prompt));
    }
    if (prompts.empty()) {
        spdlog::error("No prompts provided");
        easy_gpt::print_usage(std::cerr);
        return 1;
    }
    for (string& prompt : prompts) {
        prompt = easy_gpt::apply_chat_template(prompt);
    }

    std::string output_path;
    if (!options.prompt_file.empty()) {
        std::filesystem::path prompt_path(options.prompt_file);
        const std::string stem = prompt_path.stem().string();
        const std::string ext = prompt_path.extension().string();
        const std::string output_name = stem + "_output" + ext;
        output_path = (prompt_path.parent_path() / output_name).string();
    }

    auto config = make_unique<easy_gpt::Config>();
    config->load_config();
    config->max_steps = options.max_steps;
    config->temperature = options.temperature;
    config->top_p = options.top_p;
    config->top_k = options.top_k;
    config->seed = options.seed;
    config->use_greedy = options.use_greedy;
    auto model_param = easy_gpt::ModelParam::load(config->model_path);
    auto tokenizer = easy_gpt::Tokenizer::create(*config);
    auto data_manager = make_unique<easy_gpt::DataManager>(move(tokenizer));
    auto gpt_model = easy_gpt::GptModel::create(*config, *data_manager, *model_param);
    easy_gpt::GptEngine gpt_engine{move(gpt_model), move(config), move(data_manager)};
    gpt_engine.run(prompts, output_path);
    spdlog::info("end gpt_engine running");
    return 0;
}
