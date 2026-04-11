#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void split_f32(float *out_a_ptr, float *out_b_ptr, float *in_ptr,
                           int n_a, int n_b) {
  const int NA = n_a < 0 ? 0 : n_a;
  const int NB = n_b < 0 ? 0 : n_b;
  for (int i = 0; i < NA; ++i)
    out_a_ptr[i] = in_ptr[i];
  for (int i = 0; i < NB; ++i)
    out_b_ptr[i] = in_ptr[NA + i];
}
