#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

constexpr int kM = PTO_QEMU_SMOKE ? 16 : 64;
constexpr int kN = PTO_QEMU_SMOKE ? 16 : 64;
constexpr int kK = PTO_QEMU_SMOKE ? 16 : 64;
constexpr int kBM = 8;
constexpr int kBN = 8;
constexpr int kBK = 8;

} // namespace

extern "C" void gemm_reuse_ab_f16(fp16_t *a_ptr, fp16_t *b_ptr, fp16_t *c_ptr) {
  static float a[kM * kK];
  static float b[kK * kN];
  static float c[kM * kN];

  kernels::lowp_to_float(a_ptr, a, kM * kK);
  kernels::lowp_to_float(b_ptr, b, kK * kN);

  for (int i = 0; i < kM * kN; ++i)
    c[i] = 0.0f;

  kernels::tile_touch<float>(a);

  for (int i0 = 0; i0 < kM; i0 += kBM) {
    for (int j0 = 0; j0 < kN; j0 += kBN) {
      for (int k0 = 0; k0 < kK; k0 += kBK) {
        const int i1 = i0 + kBM > kM ? kM : i0 + kBM;
        const int j1 = j0 + kBN > kN ? kN : j0 + kBN;
        const int k1 = k0 + kBK > kK ? kK : k0 + kBK;

        for (int i = i0; i < i1; ++i) {
          for (int k = k0; k < k1; ++k) {
            const float a_ik = a[i * kK + k];
            for (int j = j0; j < j1; ++j)
              c[i * kN + j] += a_ik * b[k * kN + j];
          }
        }
      }
    }
  }

  kernels::float_to_lowp(c, c_ptr, kM * kN);
}
