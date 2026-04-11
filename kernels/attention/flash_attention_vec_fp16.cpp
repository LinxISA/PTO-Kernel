#include <common/extended_kernel_runtime.hpp>
#include <common/block_vector_kernels.hpp>
#include <common/runtime/kernel_shapes.hpp>
#include <common/runtime/kernel_tiling.hpp>

using namespace pto;

namespace {

constexpr int kS = kernels::shapes::kAttentionSeq;
constexpr int kD = kernels::shapes::kAttentionQD;

} // namespace

extern "C" void flash_attention_vec_f16(fp16_t *out_ptr, fp16_t *q_ptr,
                                         fp16_t *k_ptr, fp16_t *v_ptr) {
  static float q[kS * kD];
  static float k[kS * kD];
  static float v[kS * kD];
  static float o[kS * kD];

  kernels::lowp_to_float(q_ptr, q, kS * kD);
  kernels::lowp_to_float(k_ptr, k, kS * kD);
  kernels::lowp_to_float(v_ptr, v, kS * kD);

  kernels::tile_touch<float>(q);
#if PTO_QEMU_SMOKE
  kernels::dense_attention_f32<kS>(q, k, v, o, kS, kD, kD, false);
#elif PTO_USE_MIXED_TILE_SIMT
  kernels::mixed_attention_f32<kS, kD, kD, kernels::tiling::kFlashVecTileM,
                               kernels::tiling::kFlashVecTileK,
                               kernels::tiling::kFlashVecYDim, false>(o, q, k,
                                                                       v);
#else
  kernels::dense_attention_f32<kS>(q, k, v, o, kS, kD, kD, false);
#endif
  kernels::float_to_lowp(o, out_ptr, kS * kD);
}
