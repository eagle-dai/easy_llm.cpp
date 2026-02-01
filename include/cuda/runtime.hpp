#ifndef EASY_GPT_CUDA_RUNTIME_HPP
#define EASY_GPT_CUDA_RUNTIME_HPP

namespace easy_gpt {
namespace cuda {

bool initialize();
bool available();
void clear_weight_cache();

} // namespace cuda
} // namespace easy_gpt

#endif // EASY_GPT_CUDA_RUNTIME_HPP
