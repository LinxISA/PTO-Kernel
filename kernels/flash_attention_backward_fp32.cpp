#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

constexpr int kS = PTO_QEMU_SMOKE ? 16 : 64;
constexpr int kD = 16;

} // namespace

extern "C" void flash_attention_backward_f32(float *dq_ptr, float *dk_ptr,
                                              float *dv_ptr, float *q_ptr,
                                              float *k_ptr, float *v_ptr,
                                              float *dout_ptr) {
  kernels::tile_touch<float>(q_ptr);
  kernels::flash_backward_f32<kS>(dq_ptr, dk_ptr, dv_ptr, q_ptr, k_ptr, v_ptr,
                                  dout_ptr, kS, kD, kD);
}
