#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

constexpr int kS = PTO_QEMU_SMOKE ? 16 : 128;
constexpr int kD = 16;
constexpr int kRMax = 8;

} // namespace

extern "C" void flash_mla_deepseekv3_f8e4m3(
    fp8_e4m3_t *out_ptr, fp8_e4m3_t *q_ptr, fp8_e4m3_t *k_ptr,
    fp8_e4m3_t *v_ptr, fp8_e4m3_t *wq_ptr, fp8_e4m3_t *wk_ptr,
    fp8_e4m3_t *wv_ptr, fp8_e4m3_t *wo_ptr, int lora_rank) {
  const int rank = lora_rank < 1 ? 1 : (lora_rank > kRMax ? kRMax : lora_rank);

  static float q[kS * kD];
  static float k[kS * kD];
  static float v[kS * kD];
  static float wq[kD * kRMax];
  static float wk[kD * kRMax];
  static float wv[kD * kRMax];
  static float wo[kRMax * kD];
  static float ql[kS * kRMax];
  static float kl[kS * kRMax];
  static float vl[kS * kRMax];
  static float ctx[kS * kRMax];
  static float out[kS * kD];

  kernels::lowp_to_float(q_ptr, q, kS * kD);
  kernels::lowp_to_float(k_ptr, k, kS * kD);
  kernels::lowp_to_float(v_ptr, v, kS * kD);
  kernels::lowp_to_float(wq_ptr, wq, kD * kRMax);
  kernels::lowp_to_float(wk_ptr, wk, kD * kRMax);
  kernels::lowp_to_float(wv_ptr, wv, kD * kRMax);
  kernels::lowp_to_float(wo_ptr, wo, kRMax * kD);

  kernels::tile_touch<float>(q);

  for (int s = 0; s < kS; ++s) {
    for (int r = 0; r < rank; ++r) {
      float aq = 0.0f;
      float ak = 0.0f;
      float av = 0.0f;
      for (int d = 0; d < kD; ++d) {
        aq += q[s * kD + d] * wq[d * kRMax + r];
        ak += k[s * kD + d] * wk[d * kRMax + r];
        av += v[s * kD + d] * wv[d * kRMax + r];
      }
      ql[s * kRMax + r] = aq;
      kl[s * kRMax + r] = ak;
      vl[s * kRMax + r] = av;
    }
    for (int r = rank; r < kRMax; ++r) {
      ql[s * kRMax + r] = 0.0f;
      kl[s * kRMax + r] = 0.0f;
      vl[s * kRMax + r] = 0.0f;
    }
  }

  kernels::dense_attention_f32<kS>(ql, kl, vl, ctx, kS, rank, rank, false);

  for (int s = 0; s < kS; ++s) {
    for (int d = 0; d < kD; ++d) {
      float acc = 0.0f;
      for (int r = 0; r < rank; ++r)
        acc += ctx[s * kRMax + r] * wo[r * kD + d];
      out[s * kD + d] = acc;
    }
  }

  kernels::float_to_lowp(out, out_ptr, kS * kD);
}
