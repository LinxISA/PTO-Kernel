#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

constexpr int kM = PTO_QEMU_SMOKE ? 16 : 64;
constexpr int kN = PTO_QEMU_SMOKE ? 16 : 64;
constexpr int kK = PTO_QEMU_SMOKE ? 16 : 64;

} // namespace

extern "C" void gemm_reuse_a_f16(fp16_t *a_ptr, fp16_t *b_ptr, fp16_t *c_ptr) {
  static float a[kM * kK];
  static float b[kK * kN];
  static float c[kM * kN];

  kernels::lowp_to_float(a_ptr, a, kM * kK);
  kernels::lowp_to_float(b_ptr, b, kK * kN);

  for (int i = 0; i < kM * kN; ++i)
    c[i] = 0.0f;

  kernels::tile_touch<float>(a);

  for (int i = 0; i < kM; ++i) {
    for (int k = 0; k < kK; ++k) {
      const float a_ik = a[i * kK + k];
      for (int j = 0; j < kN; ++j)
        c[i * kN + j] += a_ik * b[k * kN + j];
    }
  }

  kernels::float_to_lowp(c, c_ptr, kM * kN);
}
