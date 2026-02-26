#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

constexpr int kQHeads = PTO_QEMU_SMOKE ? 2 : 4;
constexpr int kKVHeads = PTO_QEMU_SMOKE ? 1 : 2;
constexpr int kS = PTO_QEMU_SMOKE ? 16 : 128;
constexpr int kD = 16;

} // namespace

extern "C" void gqa_f16(fp16_t *out_ptr, fp16_t *q_ptr, fp16_t *k_ptr,
                         fp16_t *v_ptr) {
  static float q[kQHeads * kS * kD];
  static float k[kKVHeads * kS * kD];
  static float v[kKVHeads * kS * kD];
  static float o[kQHeads * kS * kD];

  kernels::lowp_to_float(q_ptr, q, kQHeads * kS * kD);
  kernels::lowp_to_float(k_ptr, k, kKVHeads * kS * kD);
  kernels::lowp_to_float(v_ptr, v, kKVHeads * kS * kD);

  kernels::tile_touch<float>(q);
  constexpr int kGroup = kQHeads / kKVHeads;
  for (int qh = 0; qh < kQHeads; ++qh) {
    const int kvh = qh / kGroup;
    kernels::dense_attention_f32<kS>(
        q + qh * kS * kD, k + kvh * kS * kD, v + kvh * kS * kD,
        o + qh * kS * kD, kS, kD, kD, false);
  }

  kernels::float_to_lowp(o, out_ptr, kQHeads * kS * kD);
}
