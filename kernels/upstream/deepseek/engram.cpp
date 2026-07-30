#include <common/deepseek_tile_intrinsics.hpp>
#include <common/deepseek_tilekernels.hpp>

extern "C" void deepseek_engram_hash_i32(int *indices,
                                          const int64_t *token_ids, int tokens,
                                          int hashes, int buckets,
                                          uint64_t seed) {
  using namespace deepseek::pto0571;
  (void)tokens;
  (void)hashes;
  VecTile<uint32_t> input;
  VecTile<uint32_t> mixed0;
  VecTile<uint32_t> shifted0;
  VecTile<uint32_t> mixed1;
  VecTile<uint32_t> multiplied;
  VecTile<uint32_t> shifted1;
  VecTile<uint32_t> mixed2;
  VecTile<uint32_t> bounded;
  load(input, reinterpret_cast<const uint32_t *>(token_ids));
  pto::TXORS(mixed0, input, static_cast<uint32_t>(seed));
  pto::TSHRS(shifted0, mixed0, 16u);
  pto::TXOR(mixed1, mixed0, shifted0);
  pto::TMULS(multiplied, mixed1, 0x7feb352du);
  pto::TSHRS(shifted1, multiplied, 15u);
  pto::TXOR(mixed2, multiplied, shifted1);
  pto::TMINS(bounded, mixed2, static_cast<uint32_t>(buckets - 1));
  store(reinterpret_cast<uint32_t *>(indices), bounded);
}

extern "C" void deepseek_engram_fused_weight_f32(
    float *output, const float *table, const int *indices,
    const float *hash_weights, int tokens, int hashes, int buckets,
    int hidden) {
  using namespace deepseek::pto0571;
  (void)tokens;
  (void)hashes;
  (void)buckets;
  (void)hidden;
  VecTile<float> table_tile;
  VecTile<int> index_tile;
  VecTile<float> gathered;
  VecTile<float> weights;
  VecTile<float> weighted;
  VecTile<float> reduced;
  load(table_tile, table);
  load(index_tile, indices);
  load(weights, hash_weights);
  pto::TGATHER(gathered, table_tile, index_tile);
  pto::TMUL(weighted, gathered, weights);
  pto::TROWSUM(reduced, weighted);
  store(output, reduced);
}

extern "C" void deepseek_engram_gate_fwd_f32(
    float *output, float *normalized, const float *input, const float *weight,
    const float *bias, int tokens, int hidden, float eps) {
  using namespace deepseek::pto0571;
  (void)tokens;
  (void)hidden;
  VecTile<float> input_tile;
  VecTile<float> normalized_tile;
  VecTile<float> weight_tile;
  VecTile<float> weighted;
  VecTile<float> logits;
  VecTile<float> bias_tile;
  VecTile<float> biased;
  VecTile<float> gate;
  load(input_tile, input);
  load(weight_tile, weight);
  load(bias_tile, bias);
  rms_normalize(normalized_tile, input_tile, eps);
  pto::TMUL(weighted, normalized_tile, weight_tile);
  pto::TROWSUM(logits, weighted);
  pto::TADD(biased, logits, bias_tile);
  sigmoid(gate, biased);
  store(normalized, normalized_tile);
  store(output, gate);
}

extern "C" void deepseek_engram_gate_bwd_f32(
    float *grad_input, float *grad_weight, float *grad_bias,
    const float *grad_output, const float *input, const float *weight,
    int tokens, int hidden, float eps) {
  using namespace deepseek::pto0571;
  (void)tokens;
  (void)hidden;
  VecTile<float> grad;
  VecTile<float> input_tile;
  VecTile<float> weight_tile;
  VecTile<float> normalized;
  VecTile<float> grad_expanded;
  VecTile<float> weighted_grad;
  VecTile<float> weight_contrib;
  VecTile<float> weight_reduced;
  VecTile<float> bias_reduced;
  load(grad, grad_output);
  load(input_tile, input);
  load(weight_tile, weight);
  rms_normalize(normalized, input_tile, eps);
  pto::TROWEXPAND(grad_expanded, grad);
  pto::TMUL(weighted_grad, grad_expanded, weight_tile);
  pto::TMUL(weight_contrib, grad_expanded, normalized);
  pto::TCOLSUM(weight_reduced, weight_contrib);
  pto::TROWSUM(bias_reduced, grad_expanded);
  store(grad_input, weighted_grad);
  store(grad_weight, weight_reduced);
  store(grad_bias, bias_reduced);
}

extern "C" void deepseek_engram_grad_w_reduce_f32(float *reduced,
                                                   const float *partial,
                                                   int parts, int rows,
                                                   int cols) {
  using namespace deepseek::pto0571;
  (void)parts;
  (void)rows;
  (void)cols;
  VecTile<float> input;
  VecTile<float> output;
  load(input, partial);
  pto::TCOLSUM(output, input);
  store(reduced, output);
}
