#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

constexpr int kMaxExperts = 256;

} // namespace

extern "C" void moe_topk_f32(float *topk_scores_ptr, int *topk_indices_ptr,
                              float *scores_ptr, int tokens, int experts,
                              int k_select) {
  const int tN = tokens < 1 ? 1 : tokens;
  const int eN = experts < 1 ? 1 : (experts > kMaxExperts ? kMaxExperts : experts);
  const int kN = k_select < 1 ? 1 : (k_select > eN ? eN : k_select);

  kernels::tile_touch<float>(scores_ptr);

  float tmp_scores[kMaxExperts];
  int tmp_idx[kMaxExperts];

  for (int t = 0; t < tN; ++t) {
    kernels::sort_desc_values_idx<kMaxExperts>(scores_ptr + t * eN, eN, tmp_scores,
                                                tmp_idx);
    for (int k = 0; k < kN; ++k) {
      topk_scores_ptr[t * kN + k] = tmp_scores[k];
      topk_indices_ptr[t * kN + k] = tmp_idx[k];
    }
  }
}
