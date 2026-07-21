#include <common/deepseek_tilekernels.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kRows = 32;
constexpr int kCols = 32;
constexpr int kElements = kRows * kCols;

alignas(64) float lhs[kElements];
alignas(64) float rhs[kElements];
alignas(64) float scales[kElements];
alignas(64) float dequantized[kElements];
alignas(64) int8_t quantized[kElements];

bool check_per_channel_fused() {
  for (int i = 0; i < kElements; ++i) {
    lhs[i] = static_cast<float>((i % 17) - 8) * 0.25f;
    rhs[i] = static_cast<float>((i % 13) + 1) * 0.0625f;
    scales[i] = 0.0f;
    dequantized[i] = 0.0f;
    quantized[i] = 0;
  }
  deepseek_quant_per_channel_fused_i8(quantized, scales, lhs, rhs, kRows,
                                      kCols);
  deepseek_quant_cast_back_f32(dequantized, quantized, scales, kRows, kCols,
                               false);
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols; ++c) {
      const int i = r * kCols + c;
      const float expected = lhs[i] + rhs[i];
      const float reconstructed = dequantized[i];
      if (!(scales[c] > 0.0f) || std::fabs(reconstructed - expected) >= 0.03f) {
        std::fprintf(stderr,
                     "per-channel fused mismatch at %d: expected=%g actual=%g "
                     "q=%d scale=%g\n",
                     i, expected, reconstructed, static_cast<int>(quantized[i]),
                     scales[c]);
        return false;
      }
    }
  }
  return true;
}

} // namespace

int main() { return check_per_channel_fused() ? 0 : 1; }
