#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

constexpr int kMaxCols = 512;

} // namespace

extern "C" void softmax_f32(float *out_ptr, float *x_ptr, int rows, int cols) {
  const int R = rows < 1 ? 1 : rows;
  const int C = cols < 1 ? 1 : (cols > kMaxCols ? kMaxCols : cols);

  float row[kMaxCols];
  for (int r = 0; r < R; ++r) {
    for (int c = 0; c < C; ++c)
      row[c] = x_ptr[r * C + c];
    kernels::softmax_inplace<kMaxCols>(row, C);
    for (int c = 0; c < C; ++c)
      out_ptr[r * C + c] = row[c];
  }
}
