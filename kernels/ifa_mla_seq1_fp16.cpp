#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

constexpr int kMaxS = PTO_QEMU_SMOKE ? 16 : 128;
constexpr int kD = 16;
constexpr int kRMax = 8;

} // namespace

extern "C" void ifa_mla_seq1_f16(fp16_t *out_ptr, fp16_t *q_ptr,
                                  fp16_t *k_cache_ptr, fp16_t *v_cache_ptr,
                                  fp16_t *wk_ptr, fp16_t *wv_ptr,
                                  int seq_len, int lora_rank) {
  const int S = seq_len < 1 ? 1 : (seq_len > kMaxS ? kMaxS : seq_len);
  const int rank = lora_rank < 1 ? 1 : (lora_rank > kRMax ? kRMax : lora_rank);

  static float q[kD];
  static float kcache[kMaxS * kD];
  static float vcache[kMaxS * kD];
  static float wk[kD * kRMax];
  static float wv[kD * kRMax];
  static float ql[kRMax];
  static float kl[kMaxS * kRMax];
  static float vl[kMaxS * kRMax];
  static float scores[kMaxS];
  static float out_lat[kRMax];
  static float out[kD];

  kernels::lowp_to_float(q_ptr, q, kD);
  kernels::lowp_to_float(k_cache_ptr, kcache, kMaxS * kD);
  kernels::lowp_to_float(v_cache_ptr, vcache, kMaxS * kD);
  kernels::lowp_to_float(wk_ptr, wk, kD * kRMax);
  kernels::lowp_to_float(wv_ptr, wv, kD * kRMax);

  kernels::tile_touch<float>(kcache);

  for (int r = 0; r < rank; ++r) {
    float acc = 0.0f;
    for (int d = 0; d < kD; ++d)
      acc += q[d] * wk[d * kRMax + r];
    ql[r] = acc;
  }

  for (int s = 0; s < S; ++s) {
    for (int r = 0; r < rank; ++r) {
      float ak = 0.0f;
      float av = 0.0f;
      for (int d = 0; d < kD; ++d) {
        ak += kcache[s * kD + d] * wk[d * kRMax + r];
        av += vcache[s * kD + d] * wv[d * kRMax + r];
      }
      kl[s * kRMax + r] = ak;
      vl[s * kRMax + r] = av;
    }
  }

  const float scale = 1.0f / kernels::m_sqrt(static_cast<float>(rank));
  for (int s = 0; s < S; ++s) {
    scores[s] = kernels::dot_f32(ql, kl + s * kRMax, rank) * scale;
  }
  kernels::softmax_inplace<kMaxS>(scores, S);

  for (int r = 0; r < rank; ++r) {
    float acc = 0.0f;
    for (int s = 0; s < S; ++s)
      acc += scores[s] * vl[s * kRMax + r];
    out_lat[r] = acc;
  }

  for (int d = 0; d < kD; ++d) {
    float acc = 0.0f;
    for (int r = 0; r < rank; ++r)
      acc += out_lat[r] * wk[d * kRMax + r];
    out[d] = acc;
  }

  kernels::float_to_lowp(out, out_ptr, kD);
}
