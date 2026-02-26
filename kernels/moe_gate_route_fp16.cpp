#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

constexpr int kMaxExperts = 256;

} // namespace

extern "C" void moe_gate_route_f16(int *routes_ptr, fp16_t *route_scores_ptr,
                                    fp16_t *logits_ptr, int tokens,
                                    int experts, int k_select) {
  const int tN = tokens < 1 ? 1 : tokens;
  const int eN = experts < 1 ? 1 : (experts > kMaxExperts ? kMaxExperts : experts);
  const int kN = k_select < 1 ? 1 : (k_select > eN ? eN : k_select);

  static float logits[kMaxExperts];
  float sorted[kMaxExperts];
  int sorted_idx[kMaxExperts];

  kernels::tile_touch<fp16_t>(logits_ptr);

  for (int t = 0; t < tN; ++t) {
    kernels::lowp_to_float(logits_ptr + t * eN, logits, eN);
    kernels::softmax_inplace<kMaxExperts>(logits, eN);
    kernels::sort_desc_values_idx<kMaxExperts>(logits, eN, sorted, sorted_idx);

    for (int k = 0; k < kN; ++k) {
      routes_ptr[t * kN + k] = sorted_idx[k];
      route_scores_ptr[t * kN + k] = float_to_fp16(sorted[k]);
    }
  }
}
