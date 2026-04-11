#include <common/extended_kernel_runtime.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

constexpr int kS = PTO_QEMU_SMOKE ? 16 : 128;
constexpr int kQHeads = PTO_QEMU_SMOKE ? 2 : 4;
constexpr int kKVHeads = PTO_QEMU_SMOKE ? 1 : 2;
constexpr int kD = 16;
constexpr int kMaxPages = PTO_QEMU_SMOKE ? 8 : 64;
constexpr int kMaxPageSize = 64;
constexpr int kMaxTokens = kMaxPages * kMaxPageSize;

} // namespace

extern "C" void paged_attention_gqa_f16(fp16_t *out_ptr, fp16_t *q_ptr,
                                         fp16_t *k_pages_ptr,
                                         fp16_t *v_pages_ptr,
                                         int *page_table_ptr, int num_pages,
                                         int page_size) {
  const int np = num_pages < 1 ? 1 : (num_pages > kMaxPages ? kMaxPages : num_pages);
  const int ps = page_size < 1 ? 1 : (page_size > kMaxPageSize ? kMaxPageSize : page_size);
  const int tokens = np * ps;

  static float q[kQHeads * kS * kD];
  static float k_pages[kKVHeads * kMaxTokens * kD];
  static float v_pages[kKVHeads * kMaxTokens * kD];
  static float k_gather[kKVHeads * kMaxTokens * kD];
  static float v_gather[kKVHeads * kMaxTokens * kD];
  static float out[kQHeads * kS * kD];

  kernels::lowp_to_float(q_ptr, q, kQHeads * kS * kD);
  kernels::lowp_to_float(k_pages_ptr, k_pages, kKVHeads * kMaxTokens * kD);
  kernels::lowp_to_float(v_pages_ptr, v_pages, kKVHeads * kMaxTokens * kD);

  for (int h = 0; h < kKVHeads; ++h) {
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
        k_gather[h * kMaxTokens * kD + t * kD + d] =
            k_pages[h * kMaxTokens * kD + physical_token * kD + d];
        v_gather[h * kMaxTokens * kD + t * kD + d] =
            v_pages[h * kMaxTokens * kD + physical_token * kD + d];
      }
    }
  }

  kernels::tile_touch<float>(q);
  constexpr int kGroup = kQHeads / kKVHeads;
  for (int qh = 0; qh < kQHeads; ++qh) {
    const int kvh = qh / kGroup;
    kernels::dense_attention_f32<kMaxTokens>(
        q + qh * kS * kD, k_gather + kvh * kMaxTokens * kD,
        v_gather + kvh * kMaxTokens * kD, out + qh * kS * kD, kS, kD, kD,
        false);
  }

  kernels::float_to_lowp(out, out_ptr, kQHeads * kS * kD);
}
