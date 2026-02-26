#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void concat_f32(float *out_ptr, float *a_ptr, float *b_ptr, int n_a,
                            int n_b) {
  const int NA = n_a < 0 ? 0 : n_a;
  const int NB = n_b < 0 ? 0 : n_b;
  for (int i = 0; i < NA; ++i)
    out_ptr[i] = a_ptr[i];
  for (int i = 0; i < NB; ++i)
    out_ptr[NA + i] = b_ptr[i];
}
