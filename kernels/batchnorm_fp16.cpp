#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void batchnorm_f16(fp16_t *out_ptr, fp16_t *x_ptr, fp16_t *mean_ptr,
                               fp16_t *var_ptr, fp16_t *gamma_ptr,
                               fp16_t *beta_ptr, float eps, int n, int c) {
  const int N = n < 1 ? 1 : n;
  const int C = c < 1 ? 1 : c;
  const float e = eps > 0.0f ? eps : 1e-5f;

  for (int i = 0; i < N; ++i) {
    for (int ch = 0; ch < C; ++ch) {
      const float x = fp16_to_float(x_ptr[i * C + ch]);
      const float m = fp16_to_float(mean_ptr[ch]);
      const float v = fp16_to_float(var_ptr[ch]);
      const float g = fp16_to_float(gamma_ptr[ch]);
      const float b = fp16_to_float(beta_ptr[ch]);
      const float y = ((x - m) / kernels::m_sqrt(v + e)) * g + b;
      out_ptr[i * C + ch] = float_to_fp16(y);
    }
  }
}
