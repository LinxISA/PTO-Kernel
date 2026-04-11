#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

constexpr int kS = PTO_QEMU_SMOKE ? 16 : 128;
constexpr int kN = kS * kS;

} // namespace

extern "C" void attention_dropout_f16(fp16_t *dst_ptr, fp16_t *src_ptr,
                                       float keep_prob, uint64_t seed) {
  static float src[kN];
  static float dst[kN];

  kernels::lowp_to_float(src_ptr, src, kN);
  kernels::tile_touch<float>(src);
  kernels::dropout_f32(dst, src, kN, keep_prob, seed);
  kernels::float_to_lowp(dst, dst_ptr, kN);
}
