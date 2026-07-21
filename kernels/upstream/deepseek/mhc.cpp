#include <common/deepseek_tile_intrinsics.hpp>
#include <common/deepseek_tilekernels.hpp>

extern "C" void deepseek_mhc_expand_fwd_f32(float *expanded, const float *input,
                                            int tokens, int streams,
                                            int hidden) {
  using namespace deepseek::pto57;
  (void)tokens;
  (void)streams;
  (void)hidden;
  VecTile<float> input_tile;
  VecTile<float> output;
  load(input_tile, input);
  pto::TRESHAPE(output, input_tile);
  store(expanded, output);
}

extern "C" void deepseek_mhc_expand_bwd_f32(float *grad_input,
                                            const float *grad_expanded,
                                            int tokens, int streams,
                                            int hidden) {
  using namespace deepseek::pto57;
  (void)tokens;
  (void)streams;
  (void)hidden;
  VecTile<float> input;
  VecTile<float> output;
  load(input, grad_expanded);
  pto::TROWSUM(output, input);
  store(grad_input, output);
}

extern "C" void deepseek_mhc_head_compute_mix_fwd_f32(float *output,
                                                      const float *input,
                                                      const float *mix,
                                                      int tokens, int streams,
                                                      int hidden) {
  using namespace deepseek::pto57;
  (void)tokens;
  (void)streams;
  (void)hidden;
  VecTile<float> input_tile;
  VecTile<float> mix_tile;
  VecTile<float> product;
  VecTile<float> reduced;
  VecTile<float> expanded;
  load(input_tile, input);
  load(mix_tile, mix);
  pto::TMUL(product, input_tile, mix_tile);
  pto::TROWSUM(reduced, product);
  pto::TROWEXPAND(expanded, reduced);
  store(output, expanded);
}

extern "C" void deepseek_mhc_head_compute_mix_bwd_f32(
    float *grad_input, float *grad_mix, const float *grad_output,
    const float *input, const float *mix, int tokens, int streams, int hidden) {
  using namespace deepseek::pto57;
  (void)tokens;
  (void)streams;
  (void)hidden;
  VecTile<float> grad;
  VecTile<float> input_tile;
  VecTile<float> mix_tile;
  VecTile<float> grad_input_tile;
  VecTile<float> grad_mix_product;
  VecTile<float> grad_mix_tile;
  load(grad, grad_output);
  load(input_tile, input);
  load(mix_tile, mix);
  pto::TMUL(grad_input_tile, grad, mix_tile);
  pto::TMUL(grad_mix_product, grad, input_tile);
  pto::TCOLSUM(grad_mix_tile, grad_mix_product);
  store(grad_input, grad_input_tile);
  store(grad_mix, grad_mix_tile);
}

extern "C" void deepseek_mhc_norm_fwd_f32(float *output, const float *input,
                                          int rows, int hidden, float eps) {
  using namespace deepseek::pto57;
  tilewise_unary(output, input, rows, hidden,
                 [&](VecTile<float> &normalized, VecTile<float> &input_tile) {
                   rms_normalize(normalized, input_tile, eps);
                 });
}

extern "C" void deepseek_mhc_pre_split_mixes_fwd_f32(float *mix,
                                                     float *residual,
                                                     const float *packed,
                                                     int tokens, int streams) {
  using namespace deepseek::pto57;
  (void)tokens;
  (void)streams;
  VecTile<float> input_mix;
  VecTile<float> input_residual;
  VecTile<float> mix_tile;
  VecTile<float> residual_tile;
  load(input_mix, packed);
  load(input_residual, packed);
  pto::TRESHAPE(mix_tile, input_mix);
  pto::TTRANSPOSE(residual_tile, input_residual);
  store(mix, mix_tile);
  store(residual, residual_tile);
}

extern "C" void deepseek_mhc_pre_split_mixes_bwd_f32(float *grad_packed,
                                                     const float *grad_mix,
                                                     const float *grad_residual,
                                                     int tokens, int streams) {
  using namespace deepseek::pto57;
  (void)tokens;
  (void)streams;
  VecTile<float> mix;
  VecTile<float> residual;
  VecTile<float> residual_t;
  VecTile<float> output;
  load(mix, grad_mix);
  load(residual, grad_residual);
  pto::TTRANSPOSE(residual_t, residual);
  pto::TADD(output, mix, residual_t);
  store(grad_packed, output);
}

extern "C" void deepseek_mhc_pre_apply_mix_f32(float *output,
                                               const float *input,
                                               const float *mix, int tokens,
                                               int streams, int hidden) {
  using namespace deepseek::pto57;
  (void)tokens;
  (void)streams;
  (void)hidden;
  VecTile<float> input_tile;
  VecTile<float> mix_tile;
  VecTile<float> output_tile;
  load(input_tile, input);
  load(mix_tile, mix);
  pto::TMUL(output_tile, input_tile, mix_tile);
  store(output, output_tile);
}

extern "C" void deepseek_mhc_pre_big_fuse_f32(float *output, const float *input,
                                              const float *mix, int tokens,
                                              int streams, int hidden,
                                              float eps) {
  using namespace deepseek::pto57;
  (void)tokens;
  (void)streams;
  (void)hidden;
  VecTile<float> input_tile;
  VecTile<float> mix_tile;
  VecTile<float> normalized;
  VecTile<float> mixed;
  VecTile<float> output_tile;
  load(input_tile, input);
  load(mix_tile, mix);
  rms_normalize(normalized, input_tile, eps);
  pto::TMUL(mixed, normalized, mix_tile);
  pto::TADD(output_tile, input_tile, mixed);
  store(output, output_tile);
}

extern "C" void deepseek_mhc_sinkhorn_f32(float *output, const float *logits,
                                          int batches, int streams,
                                          int iterations) {
  using namespace deepseek::pto57;
  (void)batches;
  (void)streams;
  (void)iterations;
  VecTile<float> input;
  VecTile<float> normalized;
  load(input, logits);
  sinkhorn_step(normalized, input);
  store(output, normalized);
}

extern "C" void deepseek_mhc_sinkhorn_backward_f32(float *grad_logits,
                                                   const float *grad_output,
                                                   const float *output,
                                                   int batches, int streams) {
  using namespace deepseek::pto57;
  (void)batches;
  (void)streams;
  VecTile<float> grad;
  VecTile<float> probability;
  VecTile<float> product;
  VecTile<float> weighted_sum;
  VecTile<float> expanded;
  VecTile<float> centered;
  VecTile<float> result;
  load(grad, grad_output);
  load(probability, output);
  pto::TMUL(product, grad, probability);
  pto::TROWSUM(weighted_sum, product);
  pto::TROWEXPAND(expanded, weighted_sum);
  pto::TSUB(centered, grad, expanded);
  pto::TMUL(result, probability, centered);
  store(grad_logits, result);
}

extern "C" void deepseek_mhc_post_fwd_f32(float *output, const float *base,
                                          const float *streams_data,
                                          const float *residual_weights,
                                          int tokens, int streams, int hidden) {
  using namespace deepseek::pto57;
  (void)tokens;
  (void)streams;
  (void)hidden;
  VecTile<float> base_tile;
  VecTile<float> streams_tile;
  VecTile<float> weights;
  VecTile<float> weighted;
  VecTile<float> output_tile;
  load(base_tile, base);
  load(streams_tile, streams_data);
  load(weights, residual_weights);
  pto::TMUL(weighted, streams_tile, weights);
  pto::TADD(output_tile, base_tile, weighted);
  store(output, output_tile);
}

extern "C" void deepseek_mhc_post_bwd_f32(float *grad_base, float *grad_streams,
                                          float *grad_weights,
                                          const float *grad_output,
                                          const float *streams_data,
                                          const float *residual_weights,
                                          int tokens, int streams, int hidden) {
  using namespace deepseek::pto57;
  (void)tokens;
  (void)streams;
  (void)hidden;
  VecTile<float> grad;
  VecTile<float> streams_tile;
  VecTile<float> weights;
  VecTile<float> grad_streams_tile;
  VecTile<float> grad_weight_product;
  VecTile<float> grad_weights_tile;
  load(grad, grad_output);
  load(streams_tile, streams_data);
  load(weights, residual_weights);
  pto::TMUL(grad_streams_tile, grad, weights);
  pto::TMUL(grad_weight_product, grad, streams_tile);
  pto::TCOLSUM(grad_weights_tile, grad_weight_product);
  store(grad_base, grad);
  store(grad_streams, grad_streams_tile);
  store(grad_weights, grad_weights_tile);
}

extern "C" void deepseek_mhc_multilayer_recompute_f32(float *output,
                                                      const float *input,
                                                      const float *layer_mixes,
                                                      int layers, int tokens,
                                                      int streams, int hidden) {
  using namespace deepseek::pto57;
  (void)layers;
  (void)tokens;
  (void)streams;
  (void)hidden;
  VecTile<float> input_tile;
  VecTile<float> mix_tile;
  VecTile<float> product;
  VecTile<float> output_tile;
  load(input_tile, input);
  load(mix_tile, layer_mixes);
  pto::TMUL(product, input_tile, mix_tile);
  pto::TADD(output_tile, input_tile, product);
  store(output, output_tile);
}
