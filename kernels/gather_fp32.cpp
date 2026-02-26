#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void gather_f32(float *out_ptr, float *in_ptr, int *indices_ptr,
                            int n) {
  const int N = n < 1 ? 1 : n;
  for (int i = 0; i < N; ++i) {
    const int idx = indices_ptr[i] < 0 ? 0 : indices_ptr[i];
    out_ptr[i] = in_ptr[idx];
  }
}
