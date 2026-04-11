#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void scatter_f32(float *out_ptr, float *in_ptr, int *indices_ptr,
                             float *updates_ptr, int n) {
  const int N = n < 1 ? 1 : n;
  for (int i = 0; i < N; ++i)
    out_ptr[i] = in_ptr[i];
  for (int i = 0; i < N; ++i) {
    const int idx = indices_ptr[i] < 0 ? 0 : indices_ptr[i];
    out_ptr[idx] = updates_ptr[i];
  }
}
