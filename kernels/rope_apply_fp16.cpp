#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

constexpr int kS = PTO_QEMU_SMOKE ? 16 : 128;
constexpr int kD = 16;

} // namespace

extern "C" void rope_apply_f16(fp16_t *q_ptr, fp16_t *k_ptr, int rotary_dim) {
  static float q[kS * kD];
  static float k[kS * kD];

  kernels::lowp_to_float(q_ptr, q, kS * kD);
  kernels::lowp_to_float(k_ptr, k, kS * kD);

  kernels::tile_touch<float>(q);
  kernels::apply_rope_f32(q, k, kS, kD, rotary_dim > 0 ? rotary_dim : kD);

  kernels::float_to_lowp(q, q_ptr, kS * kD);
  kernels::float_to_lowp(k, k_ptr, kS * kD);
}
