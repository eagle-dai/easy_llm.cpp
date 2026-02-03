#ifndef EASY_LLM_CUDA_RUNTIME_HPP
#define EASY_LLM_CUDA_RUNTIME_HPP

namespace easy_llm {
namespace cuda {

bool initialize();
bool available();
void clear_weight_cache();

} // namespace cuda
} // namespace easy_llm

#endif // EASY_LLM_CUDA_RUNTIME_HPP
