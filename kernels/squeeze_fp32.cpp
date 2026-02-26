#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void squeeze_f32(float *out_ptr, float *in_ptr, int n) {
  const int N = n < 1 ? 1 : n;
  for (int i = 0; i < N; ++i)
    out_ptr[i] = in_ptr[i];
}
