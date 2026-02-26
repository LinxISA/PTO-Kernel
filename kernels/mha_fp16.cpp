#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

constexpr int kHeads = PTO_QEMU_SMOKE ? 2 : 4;
constexpr int kS = PTO_QEMU_SMOKE ? 16 : 128;
constexpr int kD = 16;

} // namespace

extern "C" void mha_f16(fp16_t *out_ptr, fp16_t *q_ptr, fp16_t *k_ptr,
                         fp16_t *v_ptr) {
  static float q[kHeads * kS * kD];
  static float k[kHeads * kS * kD];
  static float v[kHeads * kS * kD];
  static float o[kHeads * kS * kD];

  kernels::lowp_to_float(q_ptr, q, kHeads * kS * kD);
  kernels::lowp_to_float(k_ptr, k, kHeads * kS * kD);
  kernels::lowp_to_float(v_ptr, v, kHeads * kS * kD);

  kernels::tile_touch<float>(q);
  for (int h = 0; h < kHeads; ++h) {
    const int off = h * kS * kD;
    kernels::dense_attention_f32<kS>(q + off, k + off, v + off, o + off, kS,
                                     kD, kD, false);
  }

  kernels::float_to_lowp(o, out_ptr, kHeads * kS * kD);
}
