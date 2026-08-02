#include <common/deepseek_tile_intrinsics.hpp>
#include <common/deepseek_tilekernels.hpp>

extern "C" void deepseek_batched_transpose_f32(float *dst, const float *src,
                                               int batches, int rows,
                                               int cols) {
  using namespace deepseek::pto0571;
  for_each_index(batches, [&](int batch) {
    for_each_tile_2d(
        rows, cols, [&](int row, int col, int valid_rows, int valid_cols) {
          VecTile<float> input(valid_rows, valid_cols);
          VecTile<float> output;
          load(input, src + batch * rows * cols + row * cols + col, valid_rows,
               valid_cols, cols);
          pto::TTRANS(output, input);
          store(dst + batch * rows * cols + col * rows + row, output, rows);
        });
  });
}
