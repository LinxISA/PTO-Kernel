#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void sigmoid_f32(float *out_ptr, float *x_ptr, int n) {
  const int N = n < 1 ? 1 : n;
  for (int i = 0; i < N; ++i) {
    const float x = kernels::clampf(x_ptr[i], -20.0f, 20.0f);
    out_ptr[i] = 1.0f / (1.0f + kernels::m_exp(-x));
  }
}
