#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

constexpr int kMaxS = PTO_QEMU_SMOKE ? 16 : 128;
constexpr int kQHeads = PTO_QEMU_SMOKE ? 2 : 4;
constexpr int kKVHeads = PTO_QEMU_SMOKE ? 1 : 2;
constexpr int kD = 16;

} // namespace

extern "C" void ifa_gqa_seq1_f16(fp16_t *out_ptr, fp16_t *q_ptr,
                                  fp16_t *k_cache_ptr, fp16_t *v_cache_ptr,
                                  int seq_len) {
  const int S = seq_len < 1 ? 1 : (seq_len > kMaxS ? kMaxS : seq_len);

  static float q[kQHeads * kD];
  static float kcache[kKVHeads * kMaxS * kD];
  static float vcache[kKVHeads * kMaxS * kD];
  static float scores[kMaxS];
  static float out[kQHeads * kD];

  kernels::lowp_to_float(q_ptr, q, kQHeads * kD);
  kernels::lowp_to_float(k_cache_ptr, kcache, kKVHeads * kMaxS * kD);
  kernels::lowp_to_float(v_cache_ptr, vcache, kKVHeads * kMaxS * kD);

  kernels::tile_touch<float>(kcache);

  constexpr int kGroup = kQHeads / kKVHeads;
  const float scale = 1.0f / kernels::m_sqrt(static_cast<float>(kD));

  for (int qh = 0; qh < kQHeads; ++qh) {
    const int kvh = qh / kGroup;
    const float *qv = q + qh * kD;
    const float *kh = kcache + kvh * kMaxS * kD;
    const float *vh = vcache + kvh * kMaxS * kD;

    for (int s = 0; s < S; ++s)
      scores[s] = kernels::dot_f32(qv, kh + s * kD, kD) * scale;
    kernels::softmax_inplace<kMaxS>(scores, S);

    for (int d = 0; d < kD; ++d) {
      float acc = 0.0f;
      for (int s = 0; s < S; ++s)
        acc += scores[s] * vh[s * kD + d];
      out[qh * kD + d] = acc;
    }
  }

  kernels::float_to_lowp(out, out_ptr, kQHeads * kD);
}
