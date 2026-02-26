#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

constexpr int kMaxVocab = 2048;

} // namespace

extern "C" void decode_topp_nucleus_filter_f32(float *filtered_ptr,
                                                 float *logits_ptr, int n,
                                                 float top_p) {
  const int N = n < 1 ? 1 : (n > kMaxVocab ? kMaxVocab : n);
  const float P = kernels::clampf(top_p, 0.01f, 1.0f);

  float probs[kMaxVocab];
  int idx[kMaxVocab];
  for (int i = 0; i < N; ++i)
    probs[i] = logits_ptr[i];
  kernels::softmax_inplace<kMaxVocab>(probs, N);

  float sorted_scores[kMaxVocab];
  kernels::sort_desc_values_idx<kMaxVocab>(probs, N, sorted_scores, idx);

  for (int i = 0; i < N; ++i)
    filtered_ptr[i] = -1e30f;

  float cum = 0.0f;
  for (int i = 0; i < N; ++i) {
    cum += sorted_scores[i];
    filtered_ptr[idx[i]] = logits_ptr[idx[i]];
    if (cum >= P)
      break;
  }
}
