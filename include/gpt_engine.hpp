#ifndef EASY_GPT_GPTENGINE_HPP
#define EASY_GPT_GPTENGINE_HPP

#include <memory>
#include <string>
#include <vector>

#include "config.hpp"
#include "data_manager.hpp"
#include "tokenizer.hpp"
#include "models/gpt_model.hpp"

namespace easy_gpt {

class GptEngine {
public:
    GptEngine(std::unique_ptr<GptModel>&& gpt_model,
        std::unique_ptr<Config>&& config,
        std::unique_ptr<DataManager>&& data_manager);
    GptEngine(GptEngine&& other) noexcept = default;
    GptEngine& operator=(GptEngine&& other) noexcept = default;

    void run(const std::vector<std::string>& prompts, const std::string& output_path = "");

private:
    std::unique_ptr<Config> config_;
    std::unique_ptr<DataManager> data_manager_;
    std::unique_ptr<GptModel> model_;
};

} // namespace easy_gpt

#endif // EASY_GPT_GPTENGINE_HPP
