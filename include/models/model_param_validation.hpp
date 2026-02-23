#ifndef EASY_LLM_MODELS_MODEL_PARAM_VALIDATION_HPP
#define EASY_LLM_MODELS_MODEL_PARAM_VALIDATION_HPP

namespace easy_llm {

struct Config;
class LayerKeyPrefix;
class ModelParam;

void validate_model_params_before_load(const Config& config,
                                       const LayerKeyPrefix& key_prefix,
                                       const ModelParam& model_param);
void validate_no_remaining_model_params(const ModelParam& model_param);

} // namespace easy_llm

#endif // EASY_LLM_MODELS_MODEL_PARAM_VALIDATION_HPP
