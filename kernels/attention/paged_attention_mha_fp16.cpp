#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

constexpr int kS = PTO_QEMU_SMOKE ? 16 : 128;
constexpr int kD = 16;
constexpr int kMaxPages = PTO_QEMU_SMOKE ? 8 : 64;
constexpr int kMaxPageSize = 64;
constexpr int kMaxTokens = kMaxPages * kMaxPageSize;

} // namespace

extern "C" void paged_attention_mha_f16(fp16_t *out_ptr, fp16_t *q_ptr,
                                         fp16_t *k_pages_ptr,
                                         fp16_t *v_pages_ptr,
                                         int *page_table_ptr, int num_pages,
                                         int page_size) {
  const int np = num_pages < 1 ? 1 : (num_pages > kMaxPages ? kMaxPages : num_pages);
  const int ps = page_size < 1 ? 1 : (page_size > kMaxPageSize ? kMaxPageSize : page_size);
  const int tokens = np * ps;

  static float q[kS * kD];
  static float k_pages[kMaxTokens * kD];
  static float v_pages[kMaxTokens * kD];
  static float k_gather[kMaxTokens * kD];
  static float v_gather[kMaxTokens * kD];
  static float out[kS * kD];

  kernels::lowp_to_float(q_ptr, q, kS * kD);
  kernels::lowp_to_float(k_pages_ptr, k_pages, kMaxTokens * kD);
  kernels::lowp_to_float(v_pages_ptr, v_pages, kMaxTokens * kD);

  for (int t = 0; t < tokens; ++t) {
    const int logical_page = t / ps;
    const int in_page = t % ps;
    int physical_page = page_table_ptr[logical_page];
    if (physical_page < 0)
      physical_page = 0;
    if (physical_page >= np)
      physical_page = np - 1;
    const int physical_token = physical_page * ps + in_page;
    for (int d = 0; d < kD; ++d) {
      k_gather[t * kD + d] = k_pages[physical_token * kD + d];
      v_gather[t * kD + d] = v_pages[physical_token * kD + d];
    }
  }

  kernels::tile_touch<float>(q);
  kernels::dense_attention_f32<kMaxTokens>(q, k_gather, v_gather, out, kS, kD,
                                           kD, false);
  kernels::float_to_lowp(out, out_ptr, kS * kD);
}
