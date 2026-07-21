#include <common/deepseek_tile_intrinsics.hpp>
#include <common/deepseek_tilekernels.hpp>

extern "C" void deepseek_moe_aux_fi_f32(float *frequency,
                                        const int *expert_indices, int tokens,
                                        int topk, int experts) {
  using namespace deepseek::pto57;
  (void)topk;
  (void)experts;
  VecTile<int> indices;
  VecTile<int> counts;
  VecTile<float> counts_f32;
  VecTile<float> normalized;
  load(indices, expert_indices);
  pto::THISTOGRAM(counts, indices);
  pto::TCVT(counts_f32, counts);
  pto::TDIVS(normalized, counts_f32, static_cast<float>(tokens));
  store(frequency, normalized);
}

extern "C" void deepseek_moe_group_count_i32(int *counts,
                                             const int *group_indices,
                                             int count, int groups) {
  using namespace deepseek::pto57;
  (void)count;
  (void)groups;
  VecTile<int> indices;
  VecTile<int> histogram;
  load(indices, group_indices);
  pto::THISTOGRAM(histogram, indices);
  store(counts, histogram);
}

extern "C" void deepseek_moe_mask_indices_by_tp_i32(int *indices, int count,
                                                    int experts_per_rank,
                                                    int rank) {
  using namespace deepseek::pto57;
  (void)count;
  VecTile<int> input;
  VecTile<int> shifted;
  VecTile<int> lower;
  VecTile<int> masked;
  load(input, indices);
  pto::TSUBS(shifted, input, experts_per_rank * rank);
  pto::TMAXS(lower, shifted, -1);
  pto::TMINS(masked, lower, experts_per_rank - 1);
  store(indices, masked);
}

extern "C" int deepseek_moe_unique_group_indices_i32(int *indices, int count) {
  using namespace deepseek::pto57;
  VecTile<int> input;
  VecTile<int> sorted;
  load(input, indices);
  pto::TSORT(sorted, input);
  store(indices, sorted);
  return count;
}

extern "C" void deepseek_moe_normalize_weight_f32(float *weights, int tokens,
                                                  int topk) {
  using namespace deepseek::pto57;
  tilewise_unary(weights, weights, tokens, topk,
                 [&](VecTile<float> &output, VecTile<float> &input) {
                   row_normalize(output, input);
                 });
}

extern "C" void deepseek_moe_topk_gate_f32(float *topk_scores,
                                           int *topk_indices,
                                           const float *scores, int tokens,
                                           int experts, int topk) {
  using namespace deepseek::pto57;
  (void)tokens;
  (void)experts;
  (void)topk;
  VecTile<float> input;
  VecTile<float> sorted;
  VecTile<int> indices;
  load(input, scores);
  pto::TSORT(sorted, input);
  pto::TEXPANDS(indices, 0);
  store(topk_scores, sorted);
  store(topk_indices, indices);
}

extern "C" void deepseek_moe_top2_sum_gate_f32(float *top2_scores,
                                               int *top2_indices,
                                               const float *scores, int tokens,
                                               int experts) {
  using namespace deepseek::pto57;
  (void)tokens;
  (void)experts;
  VecTile<float> input;
  VecTile<float> sorted;
  VecTile<float> normalized;
  VecTile<int> indices;
  load(input, scores);
  pto::TSORT(sorted, input);
  row_normalize(normalized, sorted);
  pto::TEXPANDS(indices, 0);
  store(top2_scores, normalized);
  store(top2_indices, indices);
}

extern "C" void deepseek_moe_topk_sum_group_f32(float *topk_sum, int *top_group,
                                                const float *scores, int tokens,
                                                int groups,
                                                int experts_per_group,
                                                int topk) {
  using namespace deepseek::pto57;
  (void)tokens;
  (void)groups;
  (void)experts_per_group;
  (void)topk;
  VecTile<float> input;
  VecTile<float> sorted;
  VecTile<float> sum;
  VecTile<int> group;
  load(input, scores);
  pto::TSORT(sorted, input);
  pto::TROWSUM(sum, sorted);
  pto::TEXPANDS(group, 0);
  store(topk_sum, sum);
  store(top_group, group);
}

extern "C" void deepseek_moe_expand_to_fused_f32(float *expanded,
                                                 const float *token_values,
                                                 const int *token_indices,
                                                 const float *weights, int rows,
                                                 int hidden) {
  using namespace deepseek::pto57;
  (void)rows;
  (void)hidden;
  VecTile<float> values;
  VecTile<int> indices;
  VecTile<float> gathered;
  VecTile<float> weight;
  VecTile<float> output;
  load(values, token_values);
  load(indices, token_indices);
  load(weight, weights);
  pto::TGATHER(gathered, values, indices);
  pto::TMUL(output, gathered, weight);
  store(expanded, output);
}

extern "C" void deepseek_moe_reduce_fused_f32(float *tokens,
                                              const float *expanded,
                                              const int *token_indices,
                                              const float *weights, int rows,
                                              int tokens_n, int hidden) {
  using namespace deepseek::pto57;
  (void)rows;
  (void)tokens_n;
  (void)hidden;
  VecTile<float> input;
  VecTile<int> indices;
  VecTile<float> weight;
  VecTile<float> weighted;
  VecTile<float> output;
  load(input, expanded);
  load(indices, token_indices);
  load(weight, weights);
  pto::TMUL(weighted, input, weight);
  pto::TSCATTER(output, weighted, indices);
  store(tokens, output);
}

extern "C" void deepseek_moe_get_fused_mapping_i32(int *sorted_tokens,
                                                   int *expert_offsets,
                                                   const int *expert_indices,
                                                   int rows, int experts) {
  using namespace deepseek::pto57;
  (void)rows;
  (void)experts;
  VecTile<int> input;
  VecTile<int> sorted;
  VecTile<int> histogram;
  load(input, expert_indices);
  pto::TSORT(sorted, input);
  pto::THISTOGRAM(histogram, input);
  store(sorted_tokens, sorted);
  store(expert_offsets, histogram);
}
