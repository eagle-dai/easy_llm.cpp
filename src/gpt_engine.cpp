#include "gpt_engine.hpp"

#include "spdlog/spdlog.h"

#include "config.hpp"
#include "utils.hpp"
#include "data_manager.hpp"
#include "tokenizer.hpp"
#include "tensor.hpp"

namespace easy_gpt {

using std::vector;
using std::string;
using std::unique_ptr;
using std::move;

GptEngine::GptEngine(unique_ptr<GptModel>&& gpt_model,
         unique_ptr<Config>&& config,
         unique_ptr<DataManager>&& data_manager)
    : config_{move(config)},
      data_manager_{move(data_manager)},
      model_{move(gpt_model)} {
    spdlog::info("Gpt Engine created");
}

void GptEngine::run(const vector<string>& prompts, const string& output_path) {
    spdlog::info("Running Gpt Engine");
    for (const string& prompt : prompts) {
        InputSample input{prompt};
        data_manager_->add_input(move(input));
    }
    auto batch = data_manager_->get_inputs();
    auto output = model_->forward(batch);
    (void)output;
    data_manager_->log_outputs(output_path);
}

} // namespace easy_gpt
