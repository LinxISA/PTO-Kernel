#include <common/extended_kernel_runtime.hpp>
#include <common/block_vector_kernels.hpp>

using namespace pto;

namespace {

#ifndef PTO_QEMU_SMOKE
#define PTO_QEMU_SMOKE 0
#endif

constexpr int kS = PTO_QEMU_SMOKE ? 16 : 128;
constexpr int kD = 16;

#ifndef PTO_USE_MIXED_TILE_SIMT
#define PTO_USE_MIXED_TILE_SIMT 0
#endif

inline void deterministic_dense_attention_f32(const float *q, const float *k,
                                              const float *v, float *o, int s,
                                              int d, int vd, bool causal) {
  float scores[kS];
  const float scale = 1.0f / kernels::m_sqrt(static_cast<float>(d));

  for (int qi = 0; qi < s; ++qi) {
    const float *qv = q + qi * d;
    for (int kj = 0; kj < s; ++kj) {
      if (causal && kj > qi) {
        scores[kj] = -1e30f;
        continue;
      }
      const float *kv = k + kj * d;
      volatile float score_acc = 0.0f;
      for (int di = 0; di < d; ++di)
        score_acc += qv[di] * kv[di];
      scores[kj] = static_cast<float>(score_acc) * scale;
    }

    float max_score = scores[0];
    for (int kj = 1; kj < s; ++kj) {
      if (scores[kj] > max_score)
        max_score = scores[kj];
    }

    volatile float sum_exp = 0.0f;
    for (int kj = 0; kj < s; ++kj) {
      scores[kj] = kernels::m_exp(scores[kj] - max_score);
      sum_exp += scores[kj];
    }
    const float inv_sum = sum_exp == 0.0f ? 0.0f : (1.0f / sum_exp);
    for (int kj = 0; kj < s; ++kj)
      scores[kj] *= inv_sum;

    for (int vi = 0; vi < vd; ++vi) {
      volatile float out_acc = 0.0f;
      for (int kj = 0; kj < s; ++kj)
        out_acc += scores[kj] * v[kj * vd + vi];
      o[qi * vd + vi] = static_cast<float>(out_acc);
    }
  }
}

} // namespace

#if defined(__clang__)
#pragma clang fp contract(off)
#endif
extern "C" void flash_attention_vec_f32(float *out_ptr, float *q_ptr,
                                         float *k_ptr, float *v_ptr) {
  kernels::tile_touch<float>(q_ptr);
#if PTO_QEMU_SMOKE
  deterministic_dense_attention_f32(q_ptr, k_ptr, v_ptr, out_ptr, kS, kD, kD,
                                    false);
#elif PTO_USE_MIXED_TILE_SIMT
  kernels::mixed_attention_f32<kS, kD, kD, 8, 4, 4, false>(
      out_ptr, q_ptr, k_ptr, v_ptr);
#else
  kernels::dense_attention_f32<kS>(q_ptr, k_ptr, v_ptr, out_ptr, kS, kD, kD,
                                   false);
#endif
}
