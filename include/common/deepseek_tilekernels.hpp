#ifndef PTO_COMMON_DEEPSEEK_TILEKERNELS_HPP
#define PTO_COMMON_DEEPSEEK_TILEKERNELS_HPP

#include <stdint.h>

// C entry points for the PTO 0.57.1 ports of deepseek-ai/TileKernels.
//
// The physical carrier is a 32x32 row-major tile. Public dimensions are
// operative: kernels issue full carriers plus rectangular tail tiles using
// LB0=ValidCol, LB1=ValidRow, and LB2=physical Col. Scalar loops are limited to
// tile-grid control; element data paths remain PTO ISA tile intrinsics.
extern "C" {

void deepseek_batched_transpose_f32(float *dst, const float *src, int batches,
                                    int rows, int cols);

void deepseek_moe_aux_fi_f32(float *frequency, const int *expert_indices,
                             int tokens, int topk, int experts);
void deepseek_moe_group_count_i32(int *counts, const int *group_indices,
                                  int count, int groups);
void deepseek_moe_mask_indices_by_tp_i32(int *indices, int count,
                                         int experts_per_rank, int rank);
int deepseek_moe_unique_group_indices_i32(int *indices, int count);
void deepseek_moe_normalize_weight_f32(float *weights, int tokens, int topk);
void deepseek_moe_topk_gate_f32(float *topk_scores, int *topk_indices,
                                const float *scores, int tokens, int experts,
                                int topk);
void deepseek_moe_top2_sum_gate_f32(float *top2_scores, int *top2_indices,
                                    const float *scores, int tokens,
                                    int experts);
void deepseek_moe_topk_sum_group_f32(float *topk_sum, int *top_group,
                                     const float *scores, int tokens,
                                     int groups, int experts_per_group,
                                     int topk);
void deepseek_moe_expand_to_fused_f32(float *expanded,
                                      const float *token_values,
                                      const int *token_indices,
                                      const float *weights, int rows,
                                      int hidden);
void deepseek_moe_reduce_fused_f32(float *tokens, const float *expanded,
                                   const int *token_indices,
                                   const float *weights, int rows, int tokens_n,
                                   int hidden);
void deepseek_moe_get_fused_mapping_i32(int *sorted_tokens, int *expert_offsets,
                                        const int *expert_indices, int rows,
                                        int experts);

void deepseek_quant_per_token_i8(int8_t *dst, float *scales, const float *src,
                                 int tokens, int hidden);
void deepseek_quant_per_block_i8(int8_t *dst, float *scales, const float *src,
                                 int count, int block);
void deepseek_quant_per_block_lossless_i8(int8_t *dst, uint32_t *scale_bits,
                                          const float *src, int count,
                                          int block);
void deepseek_quant_per_channel_i8(int8_t *dst, float *scales, const float *src,
                                   int rows, int cols);
void deepseek_quant_per_channel_transpose_i8(int8_t *dst, float *scales,
                                             const float *src, int rows,
                                             int cols);
void deepseek_quant_per_channel_fused_i8(int8_t *dst, float *scales,
                                         const float *lhs, const float *rhs,
                                         int rows, int cols);
void deepseek_quant_cast_back_f32(float *dst, const int8_t *src,
                                  const float *scales, int rows, int cols,
                                  bool per_token);
void deepseek_quant_per_token_e5m6(uint16_t *dst, const float *src, int count);
void deepseek_quant_cast_back_e5m6(float *dst, const uint16_t *src, int count);
void deepseek_quant_swiglu_forward_per_token_i8(int8_t *dst, float *scales,
                                                const float *gate,
                                                const float *up, int tokens,
                                                int hidden);
void deepseek_quant_swiglu_forward_per_channel_transpose_i8(
    int8_t *dst, float *scales, const float *gate, const float *up, int tokens,
    int hidden);
void deepseek_quant_swiglu_backward_per_token_i8(
    int8_t *dgate, int8_t *dup, float *gate_scales, float *up_scales,
    const float *grad, const float *gate, const float *up, int tokens,
    int hidden);

void deepseek_engram_hash_i32(int *indices, const int64_t *token_ids,
                              int tokens, int hashes, int buckets,
                              uint64_t seed);
void deepseek_engram_fused_weight_f32(float *output, const float *table,
                                      const int *indices,
                                      const float *hash_weights, int tokens,
                                      int hashes, int buckets, int hidden);
void deepseek_engram_gate_fwd_f32(float *output, float *normalized,
                                  const float *input, const float *weight,
                                  const float *bias, int tokens, int hidden,
                                  float eps);
void deepseek_engram_gate_bwd_f32(float *grad_input, float *grad_weight,
                                  float *grad_bias, const float *grad_output,
                                  const float *input, const float *weight,
                                  int tokens, int hidden, float eps);
void deepseek_engram_grad_w_reduce_f32(float *reduced, const float *partial,
                                       int parts, int rows, int cols);

void deepseek_mhc_expand_fwd_f32(float *expanded, const float *input,
                                 int tokens, int streams, int hidden);
void deepseek_mhc_expand_bwd_f32(float *grad_input, const float *grad_expanded,
                                 int tokens, int streams, int hidden);
void deepseek_mhc_head_compute_mix_fwd_f32(float *output, const float *input,
                                           const float *mix, int tokens,
                                           int streams, int hidden);
void deepseek_mhc_head_compute_mix_bwd_f32(float *grad_input, float *grad_mix,
                                           const float *grad_output,
                                           const float *input, const float *mix,
                                           int tokens, int streams, int hidden);
void deepseek_mhc_norm_fwd_f32(float *output, const float *input, int rows,
                               int hidden, float eps);
void deepseek_mhc_pre_split_mixes_fwd_f32(float *mix, float *residual,
                                          const float *packed, int tokens,
                                          int streams);
void deepseek_mhc_pre_split_mixes_bwd_f32(float *grad_packed,
                                          const float *grad_mix,
                                          const float *grad_residual,
                                          int tokens, int streams);
void deepseek_mhc_pre_apply_mix_f32(float *output, const float *input,
                                    const float *mix, int tokens, int streams,
                                    int hidden);
void deepseek_mhc_pre_big_fuse_f32(float *output, const float *input,
                                   const float *mix, int tokens, int streams,
                                   int hidden, float eps);
void deepseek_mhc_sinkhorn_f32(float *output, const float *logits, int batches,
                               int streams, int iterations);
void deepseek_mhc_sinkhorn_backward_f32(float *grad_logits,
                                        const float *grad_output,
                                        const float *output, int batches,
                                        int streams);
void deepseek_mhc_post_fwd_f32(float *output, const float *base,
                               const float *streams_data,
                               const float *residual_weights, int tokens,
                               int streams, int hidden);
void deepseek_mhc_post_bwd_f32(float *grad_base, float *grad_streams,
                               float *grad_weights, const float *grad_output,
                               const float *streams_data,
                               const float *residual_weights, int tokens,
                               int streams, int hidden);
void deepseek_mhc_multilayer_recompute_f32(float *output, const float *input,
                                           const float *layer_mixes, int layers,
                                           int tokens, int streams, int hidden);

} // extern "C"

#endif // PTO_COMMON_DEEPSEEK_TILEKERNELS_HPP
