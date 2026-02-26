#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void relu_f32(float *out_ptr, float *x_ptr, int n) {
  const int N = n < 1 ? 1 : n;
  for (int i = 0; i < N; ++i) {
    const float x = x_ptr[i];
    out_ptr[i] = x > 0.0f ? x : 0.0f;
  }
}
