#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

constexpr int kMaxVocab = 2048;

} // namespace

extern "C" void decode_beam_search_f32(int *token_out_ptr, float *score_out_ptr,
                                         float *logits_ptr, int batch, int vocab,
                                         int beam_width) {
  const int B = batch < 1 ? 1 : batch;
  const int V = vocab < 1 ? 1 : (vocab > kMaxVocab ? kMaxVocab : vocab);
  const int K = beam_width < 1 ? 1 : (beam_width > V ? V : beam_width);

  float sorted_scores[kMaxVocab];
  int sorted_idx[kMaxVocab];

  for (int b = 0; b < B; ++b) {
    kernels::sort_desc_values_idx<kMaxVocab>(logits_ptr + b * V, V, sorted_scores,
                                             sorted_idx);
    for (int k = 0; k < K; ++k) {
      token_out_ptr[b * K + k] = sorted_idx[k];
      score_out_ptr[b * K + k] = sorted_scores[k];
    }
  }
}
