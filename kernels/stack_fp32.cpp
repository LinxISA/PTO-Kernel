#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void stack_f32(float *out_ptr, float *a_ptr, float *b_ptr, int n) {
  const int N = n < 1 ? 1 : n;
  for (int i = 0; i < N; ++i) {
    out_ptr[i] = a_ptr[i];
    out_ptr[N + i] = b_ptr[i];
  }
}
