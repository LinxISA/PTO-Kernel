#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void tanh_f32(float *out_ptr, float *x_ptr, int n) {
  const int N = n < 1 ? 1 : n;
  for (int i = 0; i < N; ++i) {
    const float x = kernels::clampf(x_ptr[i], -10.0f, 10.0f);
    const float e2x = kernels::m_exp(2.0f * x);
    out_ptr[i] = (e2x - 1.0f) / (e2x + 1.0f);
  }
}
