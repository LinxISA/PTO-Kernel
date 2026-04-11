#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void unique_i32(int *out_values_ptr, int *out_count_ptr,
                            int *in_values_ptr, int n) {
  const int N = n < 1 ? 1 : n;
  int unique_n = 0;
  for (int i = 0; i < N; ++i) {
    const int v = in_values_ptr[i];
    bool seen = false;
    for (int j = 0; j < unique_n; ++j) {
      if (out_values_ptr[j] == v) {
        seen = true;
        break;
      }
    }
    if (!seen)
      out_values_ptr[unique_n++] = v;
  }
  out_count_ptr[0] = unique_n;
}
