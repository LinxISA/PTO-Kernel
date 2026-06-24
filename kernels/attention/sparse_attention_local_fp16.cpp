#include <common/extended_kernel_runtime.hpp>
#include <common/block_vector_kernels.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

#ifndef PTO_SPARSE_LOCAL_SMOKE_SEQ
#define PTO_SPARSE_LOCAL_SMOKE_SEQ 16
#endif

#ifndef PTO_SPARSE_LOCAL_SMOKE_DIM
#define PTO_SPARSE_LOCAL_SMOKE_DIM 16
#endif

constexpr int kS = PTO_QEMU_SMOKE ? PTO_SPARSE_LOCAL_SMOKE_SEQ : 128;
constexpr int kD = PTO_QEMU_SMOKE ? PTO_SPARSE_LOCAL_SMOKE_DIM : 16;
static_assert(kS > 0, "sparse local attention requires at least one token");
static_assert(kD > 0, "sparse local attention requires at least one channel");

#ifndef PTO_USE_MIXED_TILE_SIMT
#define PTO_USE_MIXED_TILE_SIMT 0
#endif

} // namespace

extern "C" void sparse_attention_local_f16(fp16_t *out_ptr, fp16_t *q_ptr,
                                            fp16_t *k_ptr, fp16_t *v_ptr,
                                            int window) {
  static float q[kS * kD];
  static float k[kS * kD];
  static float v[kS * kD];
  static float o[kS * kD];

  kernels::lowp_to_float(q_ptr, q, kS * kD);
  kernels::lowp_to_float(k_ptr, k, kS * kD);
  kernels::lowp_to_float(v_ptr, v, kS * kD);

  kernels::tile_touch<float>(q);
#if PTO_QEMU_SMOKE
  kernels::sparse_attention_local_f32<kS>(q, k, v, o, kS, kD, kD,
                                          window < 0 ? 0 : window);
#elif PTO_USE_MIXED_TILE_SIMT
  kernels::mixed_attention_f32<kS, kD, kD, 16, 16, 1, false>(
      o, q, k, v, window < 0 ? 0 : window);
#else
  kernels::sparse_attention_local_f32<kS>(q, k, v, o, kS, kD, kD,
                                          window < 0 ? 0 : window);
#endif
  kernels::float_to_lowp(o, out_ptr, kS * kD);
}
