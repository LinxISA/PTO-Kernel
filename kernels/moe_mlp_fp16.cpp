#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

constexpr int kTokens = PTO_QEMU_SMOKE ? 8 : 64;
constexpr int kIn = 16;
constexpr int kHidden = 32;

} // namespace

extern "C" void moe_mlp_f16(fp16_t *out_ptr, fp16_t *in_ptr, fp16_t *w1_ptr,
                             fp16_t *w2_ptr) {
  static float in[kTokens * kIn];
  static float w1[kIn * kHidden];
  static float w2[kHidden * kIn];
  static float hid[kTokens * kHidden];
  static float out[kTokens * kIn];

  kernels::lowp_to_float(in_ptr, in, kTokens * kIn);
  kernels::lowp_to_float(w1_ptr, w1, kIn * kHidden);
  kernels::lowp_to_float(w2_ptr, w2, kHidden * kIn);

  kernels::tile_touch<float>(in);

  for (int t = 0; t < kTokens; ++t) {
    for (int h = 0; h < kHidden; ++h) {
      float acc = 0.0f;
      for (int d = 0; d < kIn; ++d)
        acc += in[t * kIn + d] * w1[d * kHidden + h];
      hid[t * kHidden + h] = acc > 0.0f ? acc : 0.0f;
    }
  }

  for (int t = 0; t < kTokens; ++t) {
    for (int d = 0; d < kIn; ++d) {
      float acc = 0.0f;
      for (int h = 0; h < kHidden; ++h)
        acc += hid[t * kHidden + h] * w2[h * kIn + d];
      out[t * kIn + d] = acc;
    }
  }

  kernels::float_to_lowp(out, out_ptr, kTokens * kIn);
}
