#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void gelu_f32(float *out_ptr, float *x_ptr, int n) {
  const int N = n < 1 ? 1 : n;
  constexpr float kSqrt2OverPi = 0.7978845608f;
  for (int i = 0; i < N; ++i) {
    const float x = x_ptr[i];
    const float x3 = x * x * x;
    const float u = kSqrt2OverPi * (x + 0.044715f * x3);
    const float t = kernels::clampf(u, -10.0f, 10.0f);
    const float et = kernels::m_exp(2.0f * t);
    const float th = (et - 1.0f) / (et + 1.0f);
    out_ptr[i] = 0.5f * x * (1.0f + th);
  }
}
