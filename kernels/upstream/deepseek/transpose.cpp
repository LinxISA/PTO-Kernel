#include <common/deepseek_tile_intrinsics.hpp>
#include <common/deepseek_tilekernels.hpp>

extern "C" void deepseek_batched_transpose_f32(float *dst, const float *src,
                                                int batches, int rows,
                                                int cols) {
  using namespace deepseek::pto57;
  (void)batches;
  (void)rows;
  (void)cols;
  VecTile<float> input;
  VecTile<float> output;
  load(input, src);
  pto::TTRANSPOSE(output, input);
  store(dst, output);
}
