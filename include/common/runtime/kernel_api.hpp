#ifndef PTO_COMMON_RUNTIME_KERNEL_API_HPP
#define PTO_COMMON_RUNTIME_KERNEL_API_HPP

#include <common/linx_lowp_types.hpp>
#include <common/runtime/kernel_env.hpp>
#include <common/runtime/kernel_shapes.hpp>
#include <common/runtime/kernel_tiling.hpp>

enum class pto_dtype {
  invalid,
  i32,
  f32,
  f16,
  f8e4m3,
  f4e2m1,
};

struct pto_tiling_config {
  int tile_m;
  int tile_n;
  int tile_k;
  int y_dim;
  int x_groups;
  int y_groups;
  int flags;
};

namespace presets {

inline constexpr pto_tiling_config kNoTiling{0, 0, 0, 0, 0, 0, 0};
inline constexpr pto_tiling_config kGemmParityTiling{
    pto::kernels::tiling::kGemmTileM,
    pto::kernels::tiling::kGemmTileN,
    pto::kernels::tiling::kGemmTileK,
    1,
    1,
    1,
    0};
inline constexpr pto_tiling_config kFlashCubeParityTiling{
    pto::kernels::tiling::kFlashCubeTileM,
    0,
    pto::kernels::tiling::kFlashCubeTileK,
    pto::kernels::tiling::kFlashCubeYDim,
    1,
    1,
    0};
inline constexpr pto_tiling_config kFlashVecParityTiling{
    pto::kernels::tiling::kFlashVecTileM,
    0,
    pto::kernels::tiling::kFlashVecTileK,
    pto::kernels::tiling::kFlashVecYDim,
    1,
    1,
    0};
inline constexpr pto_tiling_config kRmsnormParityTiling{16, 0, 16, 1, 1, 1, 0};

} // namespace presets

inline constexpr unsigned PTO_ATTN_FLAG_CAUSAL = 1u << 0;

struct pto_memory_config {
  pto_dtype dtype;
  int n;
  int rows;
  int cols;
  pto_tiling_config tiling;
};

struct pto_matmul_config {
  pto_dtype lhs_dtype;
  pto_dtype rhs_dtype;
  pto_dtype out_dtype;
  pto_dtype acc_dtype;
  int m;
  int n;
  int k;
  int repeat;
  pto_tiling_config tiling;
};

struct pto_elementwise_config {
  pto_dtype lhs_dtype;
  pto_dtype rhs_dtype;
  pto_dtype out_dtype;
  pto_dtype acc_dtype;
  int n;
  int rows;
  int cols;
  float alpha;
  float beta;
};

struct pto_attention_config {
  pto_dtype q_dtype;
  pto_dtype kv_dtype;
  pto_dtype acc_dtype;
  int seq;
  int q_dim;
  int v_dim;
  int q_heads;
  int kv_heads;
  int max_seq;
  int max_kv_seq;
  int window;
  int page_size;
  int page_count;
  int block_size;
  int repeat_passes;
  unsigned flags;
  pto_tiling_config tiling;
  float dropout_keep_prob;
  unsigned seed;
};

struct pto_normalization_config {
  pto_dtype x_dtype;
  pto_dtype scale_dtype;
  pto_dtype out_dtype;
  pto_dtype acc_dtype;
  int tokens;
  int channels;
  float eps;
  pto_tiling_config tiling;
};

struct pto_indexing_config {
  pto_dtype input_dtype;
  pto_dtype index_dtype;
  pto_dtype out_dtype;
  pto_dtype acc_dtype;
  int n;
  int rows;
  int cols;
  int num_segments;
  int start;
  int len;
  int axis;
};

struct pto_layout_config {
  pto_dtype dtype;
  int n;
  int h;
  int w;
  int c;
  int rows;
  int cols;
  int n_a;
  int n_b;
};

extern "C" {
void tload_store_i32(int *src_ptr, int *dst_ptr);
void mamulb_i32(int *lhs_ptr, int *rhs_ptr, int *dst_ptr);
void tmatmul_acc_i32(int *lhs_ptr, int *rhs_ptr, int *dst_ptr);
void gemm_i32(int *lhs_ptr, int *rhs_ptr, int *dst_ptr);
void gemm_basic_f32(float *lhs_ptr, float *rhs_ptr, float *dst_ptr);
void gemm_demo_f32(float *out_ptr, float *a_ptr, float *b_ptr);
void gemm_performance_f32(float *lhs_ptr, float *rhs_ptr, float *dst_ptr,
                          int repeat_tiles);
void add_custom_f32(float *x_ptr, float *y_ptr, float *z_ptr);
void relu_f32(float *out_ptr, float *x_ptr, int n);
void sigmoid_f32(float *out_ptr, float *x_ptr, int n);
void silu_f32(float *out_ptr, float *x_ptr, int n);
void tanh_f32(float *out_ptr, float *x_ptr, int n);
void softmax_f32(float *out_ptr, float *x_ptr, int rows, int cols);
void gelu_f32(float *out_ptr, float *x_ptr, int n);
void argmax_f32(int *idx_ptr, float *x_ptr, int rows, int cols);
void gather_f32(float *out_ptr, float *in_ptr, int *indices_ptr, int n);
void where_f32(float *out_ptr, float *cond_ptr, float *x_ptr, float *y_ptr,
               int n);
void slice_f32(float *out_ptr, float *in_ptr, int start, int len);
void scatter_f32(float *out_ptr, float *in_ptr, int *indices_ptr,
                 float *updates_ptr, int n);
void unique_i32(int *out_values_ptr, int *out_count_ptr, int *in_values_ptr,
                int n);
void unsorted_segment_sum_f32(float *out_ptr, int *segment_ids_ptr,
                              float *data_ptr, int n, int num_segments);
void concat_f32(float *out_ptr, float *a_ptr, float *b_ptr, int n_a, int n_b);
void flatten_f32(float *out_ptr, float *in_ptr, int n);
void reshape_f32(float *out_ptr, float *in_ptr, int n);
void squeeze_f32(float *out_ptr, float *in_ptr, int n);
void unsqueeze_f32(float *out_ptr, float *in_ptr, int n);
void stack_f32(float *out_ptr, float *a_ptr, float *b_ptr, int n);
void split_f32(float *out_a_ptr, float *out_b_ptr, float *in_ptr, int n_a,
               int n_b);
void permute_nhwc_nchw_f32(float *out_ptr, float *in_ptr, int n, int h, int w,
                           int c);
void transpose_large_f32(float *out_ptr, float *in_ptr, int rows, int cols);
void flash_attention_i32(int *q_ptr, int *k_ptr, int *v_ptr, int *out_ptr);
void flash_attention_demo_f32(float *out_ptr, float *q_ptr, float *k_ptr,
                              float *v_ptr);
void flash_attention_masked_f32(float *out_ptr, float *q_ptr, float *k_ptr,
                                float *v_ptr);
void fa_performance_f32(float *out_ptr, float *q_ptr, float *k_ptr,
                        float *v_ptr, int repeat_passes);
void mla_attention_demo_f32(float *out_ptr, float *q_ptr, float *k_ptr,
                            float *v_ptr, float *wq_ptr, float *wk_ptr,
                            float *wv_ptr, float *wo_ptr);
void flash_attention_cube_f16(pto::fp16_t *out_ptr, pto::fp16_t *q_ptr,
                              pto::fp16_t *k_ptr, pto::fp16_t *v_ptr);
void flash_attention_vec_f32(float *out_ptr, float *q_ptr, float *k_ptr,
                             float *v_ptr);
void flash_attention_vec_f16(pto::fp16_t *out_ptr, pto::fp16_t *q_ptr,
                             pto::fp16_t *k_ptr, pto::fp16_t *v_ptr);
void gqa_f16(pto::fp16_t *out_ptr, pto::fp16_t *q_ptr, pto::fp16_t *k_ptr,
             pto::fp16_t *v_ptr);
void sparse_attention_local_f16(pto::fp16_t *out_ptr, pto::fp16_t *q_ptr,
                                pto::fp16_t *k_ptr, pto::fp16_t *v_ptr,
                                int window);
void rmsnorm_f16(pto::fp16_t *out_ptr, pto::fp16_t *x_ptr,
                 pto::fp16_t *gamma_ptr, float eps);
}

namespace pto {
namespace kernels {

inline float kernel_api_clampf(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

inline float kernel_api_exp(float x) {
  x = kernel_api_clampf(x, -20.0f, 20.0f);
  float y = 1.0f + x * (1.0f / 256.0f);
  for (int i = 0; i < 8; ++i)
    y *= y;
  return y < 0.0f ? 0.0f : y;
}

inline float kernel_api_sqrt(float x) {
  if (x <= 0.0f)
    return 0.0f;
  float y = x > 1.0f ? x : 1.0f;
  for (int i = 0; i < 8; ++i)
    y = 0.5f * (y + x / y);
  return y;
}

inline int cfg_n(const pto_elementwise_config *cfg, int fallback = 1) {
  return cfg && cfg->n > 0 ? cfg->n : fallback;
}

inline void pto_tload_store(int *dst, int *src,
                            const pto_memory_config * = nullptr) {
  tload_store_i32(src, dst);
}

inline void pto_mamulb(int *dst, int *lhs, int *rhs,
                       const pto_matmul_config * = nullptr) {
  mamulb_i32(lhs, rhs, dst);
}

inline void pto_tmatmul_acc(int *dst, int *lhs, int *rhs,
                            const pto_matmul_config * = nullptr) {
  tmatmul_acc_i32(lhs, rhs, dst);
}

inline void pto_gemm(int *dst, int *lhs, int *rhs,
                     const pto_matmul_config * = nullptr) {
  gemm_i32(lhs, rhs, dst);
}

inline void pto_gemm_basic(float *dst, float *lhs, float *rhs,
                           const pto_matmul_config * = nullptr) {
  gemm_basic_f32(lhs, rhs, dst);
}

inline void pto_gemm_scaled(float *dst, float *lhs, float *rhs,
                            const pto_matmul_config * = nullptr) {
  gemm_demo_f32(dst, lhs, rhs);
}

inline void pto_gemm_performance(float *dst, float *lhs, float *rhs,
                                 const pto_matmul_config *cfg = nullptr) {
  gemm_performance_f32(lhs, rhs, dst, cfg && cfg->repeat > 0 ? cfg->repeat : 1);
}

inline void pto_add_custom(float *dst, float *lhs, float *rhs,
                           const pto_elementwise_config * = nullptr) {
  add_custom_f32(lhs, rhs, dst);
}

inline void pto_relu(float *dst, float *src,
                     const pto_elementwise_config *cfg = nullptr) {
  relu_f32(dst, src, cfg_n(cfg));
}

inline void pto_sigmoid(float *dst, float *src,
                        const pto_elementwise_config *cfg = nullptr) {
  sigmoid_f32(dst, src, cfg_n(cfg));
}

inline void pto_silu(float *dst, float *src,
                     const pto_elementwise_config *cfg = nullptr) {
  silu_f32(dst, src, cfg_n(cfg));
}

inline void pto_tanh(float *dst, float *src,
                     const pto_elementwise_config *cfg = nullptr) {
  tanh_f32(dst, src, cfg_n(cfg));
}

inline void pto_softmax(float *dst, float *src,
                        const pto_elementwise_config *cfg = nullptr) {
  const int rows = cfg && cfg->rows > 0 ? cfg->rows : 1;
  const int cols = cfg && cfg->cols > 0 ? cfg->cols : cfg_n(cfg);
  softmax_f32(dst, src, rows, cols);
}

inline void pto_swiglu(float *dst, float *src, float *gate,
                       const pto_elementwise_config *cfg = nullptr) {
  const int n = cfg_n(cfg);
  for (int i = 0; i < n; ++i) {
    const float g = gate[i];
    const float sg = g / (1.0f + kernel_api_exp(-kernel_api_clampf(g, -20.0f, 20.0f)));
    dst[i] = src[i] * sg;
  }
}

inline void pto_flash_attention(int *dst, int *q, int *k, int *v,
                                const pto_attention_config * = nullptr) {
  flash_attention_i32(q, k, v, dst);
}

inline void pto_flash_attention_softmax(float *dst, float *q, float *k,
                                        float *v,
                                        const pto_attention_config * = nullptr) {
  flash_attention_demo_f32(dst, q, k, v);
}

inline void pto_flash_attention_masked(float *dst, float *q, float *k,
                                       float *v,
                                       const pto_attention_config * = nullptr) {
  flash_attention_masked_f32(dst, q, k, v);
}

inline void pto_fa_performance(float *dst, float *q, float *k, float *v,
                               const pto_attention_config *cfg = nullptr) {
  fa_performance_f32(dst, q, k, v,
                     cfg && cfg->repeat_passes > 0 ? cfg->repeat_passes : 1);
}

inline void pto_mla_attention(float *dst, float *q, float *k, float *v,
                              float *wq, float *wk, float *wv, float *wo,
                              const pto_attention_config * = nullptr) {
  mla_attention_demo_f32(dst, q, k, v, wq, wk, wv, wo);
}

inline void pto_flash_attention_cube(pto::fp16_t *dst, pto::fp16_t *q,
                                     pto::fp16_t *k, pto::fp16_t *v,
                                     const pto_attention_config * = nullptr) {
  flash_attention_cube_f16(dst, q, k, v);
}

inline void pto_flash_attention_vec(float *dst, float *q, float *k, float *v,
                                    const pto_attention_config * = nullptr) {
  flash_attention_vec_f32(dst, q, k, v);
}

inline void pto_flash_attention_vec(pto::fp16_t *dst, pto::fp16_t *q,
                                    pto::fp16_t *k, pto::fp16_t *v,
                                    const pto_attention_config * = nullptr) {
  flash_attention_vec_f16(dst, q, k, v);
}

inline void pto_gqa(pto::fp16_t *dst, pto::fp16_t *q, pto::fp16_t *k,
                    pto::fp16_t *v, const pto_attention_config * = nullptr) {
  gqa_f16(dst, q, k, v);
}

inline void pto_sparse_attention_local(pto::fp16_t *dst, pto::fp16_t *q,
                                       pto::fp16_t *k, pto::fp16_t *v,
                                       const pto_attention_config *cfg = nullptr) {
  sparse_attention_local_f16(dst, q, k, v, cfg ? cfg->window : 0);
}

inline void pto_rmsnorm(pto::fp16_t *dst, pto::fp16_t *src,
                        pto::fp16_t *gamma,
                        const pto_normalization_config *cfg = nullptr) {
  rmsnorm_f16(dst, src, gamma, cfg ? cfg->eps : 1.0e-5f);
}

inline void pto_batchnorm(float *dst, float *src, float *mean, float *var,
                          float *gamma, float *beta,
                          const pto_normalization_config *cfg = nullptr) {
  const int n = cfg && cfg->tokens > 0 ? cfg->tokens : 1;
  const int c = cfg && cfg->channels > 0 ? cfg->channels : 1;
  const float eps = cfg && cfg->eps > 0.0f ? cfg->eps : 1.0e-5f;
  for (int i = 0; i < n; ++i) {
    for (int ch = 0; ch < c; ++ch) {
      const float x = src[i * c + ch];
      dst[i * c + ch] =
          ((x - mean[ch]) / kernel_api_sqrt(var[ch] + eps)) * gamma[ch] +
          beta[ch];
    }
  }
}

inline void pto_layernorm(float *dst, float *src, float *gamma, float *beta,
                          const pto_normalization_config *cfg = nullptr) {
  const int tokens = cfg && cfg->tokens > 0 ? cfg->tokens : 1;
  const int channels = cfg && cfg->channels > 0 ? cfg->channels : 1;
  const float eps = cfg && cfg->eps > 0.0f ? cfg->eps : 1.0e-5f;
  for (int t = 0; t < tokens; ++t) {
    float mean = 0.0f;
    for (int c = 0; c < channels; ++c)
      mean += src[t * channels + c];
    mean /= static_cast<float>(channels);
    float var = 0.0f;
    for (int c = 0; c < channels; ++c) {
      const float z = src[t * channels + c] - mean;
      var += z * z;
    }
    const float inv = 1.0f / kernel_api_sqrt(var / static_cast<float>(channels) + eps);
    for (int c = 0; c < channels; ++c)
      dst[t * channels + c] = (src[t * channels + c] - mean) * inv * gamma[c] + beta[c];
  }
}

inline void pto_gelu(float *dst, float *src,
                     const pto_elementwise_config *cfg = nullptr) {
  gelu_f32(dst, src, cfg_n(cfg));
}

inline void pto_argmax(int *dst, float *src,
                       const pto_indexing_config *cfg = nullptr) {
  argmax_f32(dst, src, cfg && cfg->rows > 0 ? cfg->rows : 1,
             cfg && cfg->cols > 0 ? cfg->cols : 1);
}

inline void pto_gather(float *dst, float *src, int *indices,
                       const pto_indexing_config *cfg = nullptr) {
  gather_f32(dst, src, indices, cfg && cfg->n > 0 ? cfg->n : 1);
}

inline void pto_where(float *dst, float *cond, float *lhs, float *rhs,
                      const pto_indexing_config *cfg = nullptr) {
  where_f32(dst, cond, lhs, rhs, cfg && cfg->n > 0 ? cfg->n : 1);
}

inline void pto_slice(float *dst, float *src,
                      const pto_indexing_config *cfg = nullptr) {
  slice_f32(dst, src, cfg ? cfg->start : 0, cfg && cfg->len > 0 ? cfg->len : 1);
}

inline void pto_concat(float *dst, float *lhs, float *rhs,
                       const pto_layout_config *cfg = nullptr) {
  concat_f32(dst, lhs, rhs, cfg && cfg->n_a > 0 ? cfg->n_a : 0,
             cfg && cfg->n_b > 0 ? cfg->n_b : 0);
}

inline void pto_flatten(float *dst, float *src,
                        const pto_layout_config *cfg = nullptr) {
  flatten_f32(dst, src, cfg && cfg->n > 0 ? cfg->n : 1);
}

inline void pto_reshape(float *dst, float *src,
                        const pto_layout_config *cfg = nullptr) {
  reshape_f32(dst, src, cfg && cfg->n > 0 ? cfg->n : 1);
}

inline void pto_scatter(float *dst, float *src, int *indices, float *updates,
                        const pto_indexing_config *cfg = nullptr) {
  scatter_f32(dst, src, indices, updates, cfg && cfg->n > 0 ? cfg->n : 1);
}

inline void pto_squeeze(float *dst, float *src,
                        const pto_layout_config *cfg = nullptr) {
  squeeze_f32(dst, src, cfg && cfg->n > 0 ? cfg->n : 1);
}

inline void pto_unsqueeze(float *dst, float *src,
                          const pto_layout_config *cfg = nullptr) {
  unsqueeze_f32(dst, src, cfg && cfg->n > 0 ? cfg->n : 1);
}

inline void pto_stack(float *dst, float *lhs, float *rhs,
                      const pto_layout_config *cfg = nullptr) {
  stack_f32(dst, lhs, rhs, cfg && cfg->n > 0 ? cfg->n : 1);
}

inline void pto_split(float *dst_a, float *dst_b, float *src,
                      const pto_layout_config *cfg = nullptr) {
  split_f32(dst_a, dst_b, src, cfg && cfg->n_a > 0 ? cfg->n_a : 0,
            cfg && cfg->n_b > 0 ? cfg->n_b : 0);
}

inline void pto_permute_nhwc_nchw(float *dst, float *src,
                                  const pto_layout_config *cfg = nullptr) {
  permute_nhwc_nchw_f32(dst, src, cfg && cfg->n > 0 ? cfg->n : 1,
                        cfg && cfg->h > 0 ? cfg->h : 1,
                        cfg && cfg->w > 0 ? cfg->w : 1,
                        cfg && cfg->c > 0 ? cfg->c : 1);
}

inline void pto_transpose(float *dst, float *src,
                          const pto_layout_config *cfg = nullptr) {
  transpose_large_f32(dst, src, cfg && cfg->rows > 0 ? cfg->rows : 1,
                      cfg && cfg->cols > 0 ? cfg->cols : 1);
}

inline void pto_unsorted_segment_sum(float *dst, int *segment_ids, float *data,
                                     const pto_indexing_config *cfg = nullptr) {
  unsorted_segment_sum_f32(dst, segment_ids, data, cfg && cfg->n > 0 ? cfg->n : 1,
                           cfg && cfg->num_segments > 0 ? cfg->num_segments : 1);
}

inline void pto_unique(int *dst, int *count, int *src,
                       const pto_indexing_config *cfg = nullptr) {
  unique_i32(dst, count, src, cfg && cfg->n > 0 ? cfg->n : 1);
}

} // namespace kernels
} // namespace pto

using pto::kernels::pto_add_custom;
using pto::kernels::pto_argmax;
using pto::kernels::pto_batchnorm;
using pto::kernels::pto_concat;
using pto::kernels::pto_fa_performance;
using pto::kernels::pto_flash_attention;
using pto::kernels::pto_flash_attention_cube;
using pto::kernels::pto_flash_attention_masked;
using pto::kernels::pto_flash_attention_softmax;
using pto::kernels::pto_flash_attention_vec;
using pto::kernels::pto_flatten;
using pto::kernels::pto_gather;
using pto::kernels::pto_gelu;
using pto::kernels::pto_gemm;
using pto::kernels::pto_gemm_basic;
using pto::kernels::pto_gemm_performance;
using pto::kernels::pto_gemm_scaled;
using pto::kernels::pto_gqa;
using pto::kernels::pto_layernorm;
using pto::kernels::pto_mamulb;
using pto::kernels::pto_mla_attention;
using pto::kernels::pto_permute_nhwc_nchw;
using pto::kernels::pto_relu;
using pto::kernels::pto_reshape;
using pto::kernels::pto_rmsnorm;
using pto::kernels::pto_scatter;
using pto::kernels::pto_sigmoid;
using pto::kernels::pto_silu;
using pto::kernels::pto_slice;
using pto::kernels::pto_softmax;
using pto::kernels::pto_sparse_attention_local;
using pto::kernels::pto_split;
using pto::kernels::pto_squeeze;
using pto::kernels::pto_stack;
using pto::kernels::pto_swiglu;
using pto::kernels::pto_tanh;
using pto::kernels::pto_tload_store;
using pto::kernels::pto_tmatmul_acc;
using pto::kernels::pto_transpose;
using pto::kernels::pto_unique;
using pto::kernels::pto_unsorted_segment_sum;
using pto::kernels::pto_unsqueeze;
using pto::kernels::pto_where;

#endif // PTO_COMMON_RUNTIME_KERNEL_API_HPP
