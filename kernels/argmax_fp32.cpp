#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void argmax_f32(int *idx_ptr, float *x_ptr, int rows, int cols) {
  const int R = rows < 1 ? 1 : rows;
  const int C = cols < 1 ? 1 : cols;
  for (int r = 0; r < R; ++r) {
    int best = 0;
    float bestv = x_ptr[r * C + 0];
    for (int c = 1; c < C; ++c) {
      const float v = x_ptr[r * C + c];
      if (v > bestv) {
        bestv = v;
        best = c;
      }
    }
    idx_ptr[r] = best;
  }
}
