#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

constexpr int kMaxVocab = 2048;

} // namespace

extern "C" void decode_topk_filter_f32(float *filtered_ptr, float *logits_ptr,
                                         int n, int k) {
  const int N = n < 1 ? 1 : (n > kMaxVocab ? kMaxVocab : n);
  const int K = k < 1 ? 1 : (k > N ? N : k);

  float sorted_scores[kMaxVocab];
  int sorted_idx[kMaxVocab];
  kernels::sort_desc_values_idx<kMaxVocab>(logits_ptr, N, sorted_scores,
                                           sorted_idx);

  for (int i = 0; i < N; ++i)
    filtered_ptr[i] = -1e30f;
  for (int i = 0; i < K; ++i)
    filtered_ptr[sorted_idx[i]] = logits_ptr[sorted_idx[i]];
}
