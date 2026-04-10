#include <common/extended_kernel_runtime.hpp>
#include <common/block_vector_kernels.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

constexpr int kTokens = PTO_QEMU_SMOKE ? 16 : 128;
constexpr int kD = 16;

#ifndef PTO_USE_MIXED_TILE_SIMT
#define PTO_USE_MIXED_TILE_SIMT 0
#endif

} // namespace

extern "C" void rmsnorm_f16(fp16_t *out_ptr, fp16_t *x_ptr, fp16_t *gamma_ptr,
                             float eps) {
  const float e = eps > 0.0f ? eps : 1e-5f;
  static float x[kTokens * kD];
  static float gamma[kD];
  static float out[kTokens * kD];

  kernels::lowp_to_float(x_ptr, x, kTokens * kD);
  kernels::lowp_to_float(gamma_ptr, gamma, kD);
#if PTO_QEMU_SMOKE
  for (int t = 0; t < kTokens; ++t) {
    float ss = 0.0f;
    for (int d = 0; d < kD; ++d) {
      const float xv = x[t * kD + d];
      ss += xv * xv;
    }
    const float inv = 1.0f / kernels::m_sqrt(ss / static_cast<float>(kD) + e);
    for (int d = 0; d < kD; ++d)
      out[t * kD + d] = x[t * kD + d] * gamma[d] * inv;
  }
#elif PTO_USE_MIXED_TILE_SIMT
  kernels::mixed_rmsnorm_f32<kTokens, kD>(out, x, gamma, e);
#else
  for (int t = 0; t < kTokens; ++t) {
    float ss = 0.0f;
    for (int d = 0; d < kD; ++d) {
      const float xv = x[t * kD + d];
      ss += xv * xv;
    }
    const float inv = 1.0f / kernels::m_sqrt(ss / static_cast<float>(kD) + e);
    for (int d = 0; d < kD; ++d)
      out[t * kD + d] = x[t * kD + d] * gamma[d] * inv;
  }
#endif
  kernels::float_to_lowp(out, out_ptr, kTokens * kD);
}
