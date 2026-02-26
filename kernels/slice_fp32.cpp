#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void slice_f32(float *out_ptr, float *in_ptr, int start, int len) {
  const int S = start < 0 ? 0 : start;
  const int L = len < 1 ? 1 : len;
  for (int i = 0; i < L; ++i)
    out_ptr[i] = in_ptr[S + i];
}
