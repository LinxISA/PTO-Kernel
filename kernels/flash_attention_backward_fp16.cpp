#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

constexpr int kS = PTO_QEMU_SMOKE ? 16 : 64;
constexpr int kD = 16;

} // namespace

extern "C" void flash_attention_backward_f16(fp16_t *dq_ptr, fp16_t *dk_ptr,
                                              fp16_t *dv_ptr, fp16_t *q_ptr,
                                              fp16_t *k_ptr, fp16_t *v_ptr,
                                              fp16_t *dout_ptr) {
  static float q[kS * kD];
  static float k[kS * kD];
  static float v[kS * kD];
  static float dout[kS * kD];
  static float dq[kS * kD];
  static float dk[kS * kD];
  static float dv[kS * kD];

  kernels::lowp_to_float(q_ptr, q, kS * kD);
  kernels::lowp_to_float(k_ptr, k, kS * kD);
  kernels::lowp_to_float(v_ptr, v, kS * kD);
  kernels::lowp_to_float(dout_ptr, dout, kS * kD);

  kernels::tile_touch<float>(q);
  kernels::flash_backward_f32<kS>(dq, dk, dv, q, k, v, dout, kS, kD, kD);

  kernels::float_to_lowp(dq, dq_ptr, kS * kD);
  kernels::float_to_lowp(dk, dk_ptr, kS * kD);
  kernels::float_to_lowp(dv, dv_ptr, kS * kD);
}
