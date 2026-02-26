#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

constexpr int kTokens = PTO_QEMU_SMOKE ? 16 : 128;
constexpr int kD = 16;

} // namespace

extern "C" void rmsnorm_f16(fp16_t *out_ptr, fp16_t *x_ptr, fp16_t *gamma_ptr,
                             float eps) {
  const float e = eps > 0.0f ? eps : 1e-5f;
  for (int t = 0; t < kTokens; ++t) {
    float ss = 0.0f;
    for (int d = 0; d < kD; ++d) {
      const float x = fp16_to_float(x_ptr[t * kD + d]);
      ss += x * x;
    }
    const float inv = 1.0f / kernels::m_sqrt(ss / static_cast<float>(kD) + e);
    for (int d = 0; d < kD; ++d) {
      const float x = fp16_to_float(x_ptr[t * kD + d]);
      const float g = fp16_to_float(gamma_ptr[d]);
      out_ptr[t * kD + d] = float_to_fp16(x * inv * g);
    }
  }
}
