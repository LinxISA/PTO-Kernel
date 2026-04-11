#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

constexpr int kMaxExperts = 256;

} // namespace

extern "C" void moe_sort_f32(float *sorted_scores_ptr, int *sorted_indices_ptr,
                              float *scores_ptr, int tokens, int experts) {
  const int tN = tokens < 1 ? 1 : tokens;
  const int eN = experts < 1 ? 1 : (experts > kMaxExperts ? kMaxExperts : experts);

  kernels::tile_touch<float>(scores_ptr);

  float tmp_scores[kMaxExperts];
  int tmp_idx[kMaxExperts];

  for (int t = 0; t < tN; ++t) {
    kernels::sort_desc_values_idx<kMaxExperts>(scores_ptr + t * eN, eN, tmp_scores,
                                                tmp_idx);
    for (int e = 0; e < eN; ++e) {
      sorted_scores_ptr[t * eN + e] = tmp_scores[e];
      sorted_indices_ptr[t * eN + e] = tmp_idx[e];
    }
  }
}
