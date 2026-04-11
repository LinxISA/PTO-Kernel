#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void where_f32(float *out_ptr, float *cond_ptr, float *x_ptr,
                           float *y_ptr, int n) {
  const int N = n < 1 ? 1 : n;
  for (int i = 0; i < N; ++i)
    out_ptr[i] = cond_ptr[i] > 0.0f ? x_ptr[i] : y_ptr[i];
}
