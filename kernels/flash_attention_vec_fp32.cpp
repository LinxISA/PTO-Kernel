#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

constexpr int kS = PTO_QEMU_SMOKE ? 16 : 128;
constexpr int kD = 16;

} // namespace

extern "C" void flash_attention_vec_f32(float *out_ptr, float *q_ptr,
                                         float *k_ptr, float *v_ptr) {
  kernels::tile_touch<float>(q_ptr);
  kernels::dense_attention_f32<kS>(q_ptr, k_ptr, v_ptr, out_ptr, kS, kD, kD,
                                   false);
}
