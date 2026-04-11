#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void transpose_large_f32(float *out_ptr, float *in_ptr, int rows,
                                      int cols) {
  const int R = rows < 1 ? 1 : rows;
  const int C = cols < 1 ? 1 : cols;
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < C; ++c)
      out_ptr[c * R + r] = in_ptr[r * C + c];
}
