#include <common/pto_tileop.hpp>

int main() {
  using Matrix = pto::global_tensor<float, pto::RowMajor<3, 7>>;
  using Tile = pto::Tile<pto::Location::Vec, float, 8, 8,
                         pto::BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC>;

  float input[3 * 7];
  float output[3 * 7];
  for (int i = 0; i < 3 * 7; ++i) {
    input[i] = static_cast<float>(i + 1);
    output[i] = -99.0f;
  }

  pto::global_iterator<Matrix, Tile> input_it(input);
  pto::global_iterator<Matrix, Tile> output_it(output);
  Tile lhs(3, 5);
  Tile sum(3, 5);
  pto::TLOAD(lhs, input_it(0, 0));
  pto::TADD(sum, lhs, lhs);
  pto::TSTORE(output_it(0, 0), sum);

  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 7; ++c) {
      const float expected = c < 5 ? input[r * 7 + c] * 2.0f : -99.0f;
      if (output[r * 7 + c] != expected)
        return 1 + r * 7 + c;
    }
  }

  float transposed[5 * 6];
  for (int i = 0; i < 5 * 6; ++i)
    transposed[i] = -77.0f;
  Tile transposed_tile;
  pto::TTRANSPOSE(transposed_tile, lhs);
  using TransposedMatrix = pto::global_tensor<float, pto::RowMajor<5, 6>>;
  pto::global_iterator<TransposedMatrix, Tile> transposed_it(transposed);
  pto::TSTORE(transposed_it(0, 0), transposed_tile);
  if (transposed_tile.GetValidRow() != 5 || transposed_tile.GetValidCol() != 3)
    return 100;
  for (int r = 0; r < 5; ++r) {
    for (int c = 0; c < 6; ++c) {
      const float expected = c < 3 ? input[c * 7 + r] : -77.0f;
      if (transposed[r * 6 + c] != expected)
        return 101 + r * 6 + c;
    }
  }
  return 0;
}
