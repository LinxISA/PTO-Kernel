#ifndef PTO_COMMON_EXTENDED_KERNEL_RUNTIME_HPP
#define PTO_COMMON_EXTENDED_KERNEL_RUNTIME_HPP

#include <common/dropout_rng.hpp>
#include <common/linx_lowp_types.hpp>
#include <common/pto_tileop.hpp>

#include <stdint.h>

namespace pto {
namespace kernels {

inline float m_abs(float x) { return x < 0.0f ? -x : x; }

inline float m_exp(float x) {
  if (x < -10.0f)
    return 0.0f;
  if (x > 10.0f)
    x = 10.0f;
  float term = 1.0f;
  float sum = 1.0f;
  for (int i = 1; i <= 7; ++i) {
    term *= x / static_cast<float>(i);
    sum += term;
  }
  return sum > 0.0f ? sum : 0.0f;
}

inline float m_log(float x) {
  if (x <= 0.0f)
    return -80.0f;
  float y = (x - 1.0f) / (x + 1.0f);
  float y2 = y * y;
  float acc = y;
  float p = y;
  p *= y2;
  acc += p / 3.0f;
  p *= y2;
  acc += p / 5.0f;
  p *= y2;
  acc += p / 7.0f;
  return 2.0f * acc;
}

inline float m_sqrt(float x) {
  if (x <= 0.0f)
    return 0.0f;
  float g = x > 1.0f ? x : 1.0f;
  for (int i = 0; i < 6; ++i)
    g = 0.5f * (g + x / g);
  return g;
}

inline float wrap_pi(float x) {
  constexpr float kPi = 3.14159265358979323846f;
  constexpr float kTwoPi = 6.28318530717958647692f;
  while (x > kPi)
    x -= kTwoPi;
  while (x < -kPi)
    x += kTwoPi;
  return x;
}

inline float m_sin(float x) {
  float y = wrap_pi(x);
  float y2 = y * y;
  return y * (1.0f - y2 / 6.0f + (y2 * y2) / 120.0f);
}

inline float m_cos(float x) {
  float y = wrap_pi(x);
  float y2 = y * y;
  return 1.0f - y2 / 2.0f + (y2 * y2) / 24.0f;
}

template <typename T, int Rows = 16, int Cols = 16>
inline void tile_touch(T *ptr) {
#if PTO_QEMU_SMOKE
  (void)ptr;
#else
  static_assert(Rows > 0 && Cols > 0, "Rows/Cols must be positive");
  static_assert(Rows * Cols * static_cast<int>(sizeof(T)) <= 4096,
                "tile touch must fit <=4KB");
  using gm = global_tensor<T, RowMajor<Rows, Cols>>;
  using tv = Tile<Location::Vec, T, Rows, Cols, BLayout::RowMajor>;
  using it = global_iterator<gm, tv>;
  it g(ptr);
  tv t;
  TLOAD(t, g(0, 0));
  TSTORE(g(0, 0), t);
#endif
}

inline float clampf(float x, float lo, float hi) {
  if (x < lo)
    return lo;
  if (x > hi)
    return hi;
  return x;
}

template <int MaxN>
inline void softmax_inplace(float *x, int n) {
  float m = x[0];
  for (int i = 1; i < n; ++i)
    if (x[i] > m)
      m = x[i];

  float s = 0.0f;
  for (int i = 0; i < n; ++i) {
    x[i] = m_exp(x[i] - m);
    s += x[i];
  }
  const float inv = (s == 0.0f) ? 0.0f : (1.0f / s);
  for (int i = 0; i < n; ++i)
    x[i] *= inv;
}

inline float dot_f32(const float *a, const float *b, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; ++i)
    s += a[i] * b[i];
  return s;
}

template <int MaxS>
inline void dense_attention_f32(const float *q, const float *k, const float *v,
                                float *o, int S, int D, int VD, bool causal) {
  float scores[MaxS];
  const float scale = 1.0f / m_sqrt(static_cast<float>(D));

  for (int qi = 0; qi < S; ++qi) {
    const float *qv = q + qi * D;
    for (int kj = 0; kj < S; ++kj) {
      if (causal && kj > qi) {
        scores[kj] = -1e30f;
      } else {
        const float *kv = k + kj * D;
        scores[kj] = dot_f32(qv, kv, D) * scale;
      }
    }
    softmax_inplace<MaxS>(scores, S);
    for (int d = 0; d < VD; ++d) {
      float acc = 0.0f;
      for (int kj = 0; kj < S; ++kj)
        acc += scores[kj] * v[kj * VD + d];
      o[qi * VD + d] = acc;
    }
  }
}

template <int MaxS>
inline void sparse_attention_local_f32(const float *q, const float *k,
                                       const float *v, float *o, int S, int D,
                                       int VD, int window) {
  float scores[MaxS];
  const float scale = 1.0f / m_sqrt(static_cast<float>(D));

  for (int qi = 0; qi < S; ++qi) {
    for (int kj = 0; kj < S; ++kj) {
      const int dist = qi > kj ? (qi - kj) : (kj - qi);
      if (dist > window) {
        scores[kj] = -1e30f;
      } else {
        scores[kj] = dot_f32(q + qi * D, k + kj * D, D) * scale;
      }
    }
    softmax_inplace<MaxS>(scores, S);
    for (int d = 0; d < VD; ++d) {
      float acc = 0.0f;
      for (int kj = 0; kj < S; ++kj)
        acc += scores[kj] * v[kj * VD + d];
      o[qi * VD + d] = acc;
    }
  }
}

template <int MaxS>
inline void sparse_attention_block_f32(const float *q, const float *k,
                                       const float *v, float *o, int S, int D,
                                       int VD, int block_size) {
  float scores[MaxS];
  const float scale = 1.0f / m_sqrt(static_cast<float>(D));

  for (int qi = 0; qi < S; ++qi) {
    const int qblk = qi / block_size;
    for (int kj = 0; kj < S; ++kj) {
      const int kblk = kj / block_size;
      if (kblk != qblk && kblk != qblk - 1) {
        scores[kj] = -1e30f;
      } else {
        scores[kj] = dot_f32(q + qi * D, k + kj * D, D) * scale;
      }
    }
    softmax_inplace<MaxS>(scores, S);
    for (int d = 0; d < VD; ++d) {
      float acc = 0.0f;
      for (int kj = 0; kj < S; ++kj)
        acc += scores[kj] * v[kj * VD + d];
      o[qi * VD + d] = acc;
    }
  }
}

inline void apply_rope_f32(float *q, float *k, int S, int D, int rotary_dim) {
  const int rd = rotary_dim > D ? D : rotary_dim;
  const int pairs = rd / 2;
  for (int pos = 0; pos < S; ++pos) {
    for (int i = 0; i < pairs; ++i) {
      const int a = 2 * i;
      const int b = a + 1;
      const float theta = pos * m_exp(-m_log(10000.0f) * (2.0f * i) / rd);
      const float cs = m_cos(theta);
      const float sn = m_sin(theta);

      float q0 = q[pos * D + a];
      float q1 = q[pos * D + b];
      q[pos * D + a] = q0 * cs - q1 * sn;
      q[pos * D + b] = q0 * sn + q1 * cs;

      float k0 = k[pos * D + a];
      float k1 = k[pos * D + b];
      k[pos * D + a] = k0 * cs - k1 * sn;
      k[pos * D + b] = k0 * sn + k1 * cs;
    }
  }
}

inline void dropout_f32(float *dst, const float *src, int n, float keep_prob,
                        uint64_t seed) {
  DropoutRng rng(seed);
  const float scale = keep_prob <= 0.0f ? 0.0f : (1.0f / keep_prob);
  for (int i = 0; i < n; ++i) {
    const bool keep = dropout_keep(rng, keep_prob);
    dst[i] = keep ? (src[i] * scale) : 0.0f;
  }
}

template <int MaxE>
inline void sort_desc_values_idx(const float *scores, int experts,
                                 float *sorted_scores, int *sorted_idx) {
  for (int i = 0; i < experts; ++i) {
    sorted_scores[i] = scores[i];
    sorted_idx[i] = i;
  }
  for (int i = 0; i < experts; ++i) {
    int best = i;
    for (int j = i + 1; j < experts; ++j) {
      if (sorted_scores[j] > sorted_scores[best] ||
          (sorted_scores[j] == sorted_scores[best] &&
           sorted_idx[j] < sorted_idx[best])) {
        best = j;
      }
    }
    if (best != i) {
      float vs = sorted_scores[i];
      int vi = sorted_idx[i];
      sorted_scores[i] = sorted_scores[best];
      sorted_idx[i] = sorted_idx[best];
      sorted_scores[best] = vs;
      sorted_idx[best] = vi;
    }
  }
}

template <int MaxS>
inline void flash_backward_f32(float *dq, float *dk, float *dv, const float *q,
                               const float *k, const float *v, const float *dout,
                               int S, int D, int VD) {
  float P[MaxS * MaxS];
  float dP[MaxS * MaxS];
  float dS[MaxS * MaxS];
  float row_tmp[MaxS];
  const float scale = 1.0f / m_sqrt(static_cast<float>(D));

  for (int i = 0; i < S; ++i) {
    for (int j = 0; j < S; ++j)
      row_tmp[j] = dot_f32(q + i * D, k + j * D, D) * scale;
    softmax_inplace<MaxS>(row_tmp, S);
    for (int j = 0; j < S; ++j)
      P[i * S + j] = row_tmp[j];
  }

  for (int i = 0; i < S * D; ++i) {
    dq[i] = 0.0f;
    dk[i] = 0.0f;
  }
  for (int i = 0; i < S * VD; ++i)
    dv[i] = 0.0f;

  for (int j = 0; j < S; ++j) {
    for (int d = 0; d < VD; ++d) {
      float acc = 0.0f;
      for (int i = 0; i < S; ++i)
        acc += P[i * S + j] * dout[i * VD + d];
      dv[j * VD + d] = acc;
    }
  }

  for (int i = 0; i < S; ++i) {
    for (int j = 0; j < S; ++j)
      dP[i * S + j] = dot_f32(dout + i * VD, v + j * VD, VD);
  }

  for (int i = 0; i < S; ++i) {
    float row_dot = 0.0f;
    for (int j = 0; j < S; ++j)
      row_dot += dP[i * S + j] * P[i * S + j];
    for (int j = 0; j < S; ++j)
      dS[i * S + j] = P[i * S + j] * (dP[i * S + j] - row_dot);
  }

  for (int i = 0; i < S; ++i) {
    for (int d = 0; d < D; ++d) {
      float acc = 0.0f;
      for (int j = 0; j < S; ++j)
        acc += dS[i * S + j] * k[j * D + d];
      dq[i * D + d] = acc * scale;
    }
  }

  for (int j = 0; j < S; ++j) {
    for (int d = 0; d < D; ++d) {
      float acc = 0.0f;
      for (int i = 0; i < S; ++i)
        acc += dS[i * S + j] * q[i * D + d];
      dk[j * D + d] = acc * scale;
    }
  }
}

template <typename LowpT>
inline void lowp_to_float(const LowpT *src, float *dst, int n) {
  for (int i = 0; i < n; ++i)
    dst[i] = 0.0f;
}

template <>
inline void lowp_to_float<fp16_t>(const fp16_t *src, float *dst, int n) {
  for (int i = 0; i < n; ++i)
    dst[i] = fp16_to_float(src[i]);
}

template <>
inline void lowp_to_float<fp8_e4m3_t>(const fp8_e4m3_t *src, float *dst, int n) {
  for (int i = 0; i < n; ++i)
    dst[i] = fp8_e4m3_to_float(src[i]);
}

template <>
inline void lowp_to_float<fp4_e2m1_t>(const fp4_e2m1_t *src, float *dst, int n) {
  for (int i = 0; i < n; ++i)
    dst[i] = fp4_e2m1_to_float(src[i]);
}

template <typename LowpT>
inline void float_to_lowp(const float *src, LowpT *dst, int n) {
  for (int i = 0; i < n; ++i)
    dst[i] = LowpT{};
}

template <>
inline void float_to_lowp<fp16_t>(const float *src, fp16_t *dst, int n) {
  for (int i = 0; i < n; ++i)
    dst[i] = float_to_fp16(src[i]);
}

template <>
inline void float_to_lowp<fp8_e4m3_t>(const float *src, fp8_e4m3_t *dst, int n) {
  for (int i = 0; i < n; ++i)
    dst[i] = float_to_fp8_e4m3(src[i]);
}

template <>
inline void float_to_lowp<fp4_e2m1_t>(const float *src, fp4_e2m1_t *dst, int n) {
  for (int i = 0; i < n; ++i)
    dst[i] = float_to_fp4_e2m1(src[i]);
}

} // namespace kernels
} // namespace pto

#endif // PTO_COMMON_EXTENDED_KERNEL_RUNTIME_HPP
