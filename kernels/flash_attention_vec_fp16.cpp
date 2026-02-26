#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

constexpr int kS = PTO_QEMU_SMOKE ? 16 : 128;
constexpr int kD = 16;

} // namespace

extern "C" void flash_attention_vec_f16(fp16_t *out_ptr, fp16_t *q_ptr,
                                         fp16_t *k_ptr, fp16_t *v_ptr) {
  static float q[kS * kD];
  static float k[kS * kD];
  static float v[kS * kD];
  static float o[kS * kD];

  kernels::lowp_to_float(q_ptr, q, kS * kD);
  kernels::lowp_to_float(k_ptr, k, kS * kD);
  kernels::lowp_to_float(v_ptr, v, kS * kD);

  kernels::tile_touch<float>(q);
  kernels::dense_attention_f32<kS>(q, k, v, o, kS, kD, kD, false);
  kernels::float_to_lowp(o, out_ptr, kS * kD);
}
