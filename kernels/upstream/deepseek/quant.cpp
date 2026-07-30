#include <common/deepseek_tile_intrinsics.hpp>
#include <common/deepseek_tilekernels.hpp>

extern "C" void deepseek_quant_per_token_i8(int8_t *dst, float *scales,
                                            const float *src, int tokens,
                                            int hidden) {
  using namespace deepseek::pto0571;
  for_each_tile_2d(
      tokens, hidden, [&](int row, int col, int valid_rows, int valid_cols) {
        VecTile<float> input(valid_rows, valid_cols);
        VecTile<float> scale;
        VecTile<int8_t> quantized;
        load(input, src + row * hidden + col, valid_rows, valid_cols, hidden);
        quantize_rows(quantized, scale, input, [&](VecTile<float> &ready) {
          store(scales + row, ready, 1);
        });
        store(dst + row * hidden + col, quantized, hidden);
      });
}

extern "C" void deepseek_quant_per_block_i8(int8_t *dst, float *scales,
                                            const float *src, int count,
                                            int block) {
  using namespace deepseek::pto0571;
  (void)count;
  (void)block;
  VecTile<float> input;
  VecTile<float> scale;
  VecTile<int8_t> quantized;
  load(input, src);
  quantize_rows(quantized, scale, input,
                [&](VecTile<float> &ready) { store(scales, ready, 1); });
  store(dst, quantized);
}

extern "C" void deepseek_quant_per_block_lossless_i8(int8_t *dst,
                                                     uint32_t *scale_bits,
                                                     const float *src,
                                                     int count, int block) {
  using namespace deepseek::pto0571;
  (void)count;
  (void)block;
  VecTile<float> input;
  VecTile<float> scale;
  VecTile<int8_t> quantized;
  load(input, src);
  quantize_rows(quantized, scale, input, [&](VecTile<float> &ready) {
    store(reinterpret_cast<float *>(scale_bits), ready, 1);
  });
  store(dst, quantized);
}

extern "C" void deepseek_quant_per_channel_i8(int8_t *dst, float *scales,
                                              const float *src, int rows,
                                              int cols) {
  using namespace deepseek::pto0571;
  (void)rows;
  (void)cols;
  VecTile<float> input;
  VecTile<float> transposed;
  VecTile<float> scale;
  VecTile<int8_t> quantized_t;
  VecTile<int8_t> quantized;
  load(input, src);
  pto::TTRANS(transposed, input);
  quantize_rows(quantized_t, scale, transposed,
                [&](VecTile<float> &ready) { store(scales, ready, 1); });
  pto::TTRANS(quantized, quantized_t);
  store(dst, quantized);
}

extern "C" void deepseek_quant_per_channel_transpose_i8(int8_t *dst,
                                                        float *scales,
                                                        const float *src,
                                                        int rows, int cols) {
  using namespace deepseek::pto0571;
  (void)rows;
  (void)cols;
  VecTile<float> input;
  VecTile<float> transposed;
  VecTile<float> scale;
  VecTile<int8_t> quantized;
  load(input, src);
  pto::TTRANS(transposed, input);
  quantize_rows(quantized, scale, transposed,
                [&](VecTile<float> &ready) { store(scales, ready, 1); });
  store(dst, quantized);
}

extern "C" void deepseek_quant_per_channel_fused_i8(int8_t *dst, float *scales,
                                                    const float *lhs,
                                                    const float *rhs, int rows,
                                                    int cols) {
  using namespace deepseek::pto0571;
  (void)rows;
  (void)cols;
  VecTile<float> left;
  VecTile<float> right;
  VecTile<float> fused;
  VecTile<float> transposed;
  VecTile<float> scale;
  VecTile<int8_t> quantized_t;
  VecTile<int8_t> quantized;
  load(left, lhs);
  load(right, rhs);
  pto::TADD(fused, left, right);
  pto::TTRANS(transposed, fused);
  quantize_rows(quantized_t, scale, transposed,
                [&](VecTile<float> &ready) { store(scales, ready, 1); });
  pto::TTRANS(quantized, quantized_t);
  store(dst, quantized);
}

extern "C" void deepseek_quant_cast_back_f32(float *dst, const int8_t *src,
                                             const float *scales, int rows,
                                             int cols, bool per_token) {
  using namespace deepseek::pto0571;
  for_each_tile_2d(
      rows, cols, [&](int row, int col, int valid_rows, int valid_cols) {
        VecTile<int8_t> quantized(valid_rows, valid_cols);
        VecTile<float> converted;
        VecTile<float> scale;
        VecTile<float> expanded(valid_rows, valid_cols);
        VecTile<float> output;
        load(quantized, src + row * cols + col, valid_rows, valid_cols, cols);
        pto::TCVT(converted, quantized);
        if (per_token) {
          load(scale, scales + row, valid_rows, 1, 1);
          pto::TROWEXPAND(expanded, scale);
        } else {
          load(scale, scales + col, 1, valid_cols, cols);
          pto::TCOLEXPAND(expanded, scale);
        }
        pto::TMUL(output, converted, expanded);
        store(dst + row * cols + col, output, cols);
      });
}

extern "C" void deepseek_quant_per_token_e5m6(uint16_t *dst, const float *src,
                                              int count) {
  using namespace deepseek::pto0571;
  (void)count;
  VecTile<float> input;
  VecTile<uint16_t> output;
  load(input, src);
  pto::TCVT(output, input);
  store(dst, output);
}

extern "C" void deepseek_quant_cast_back_e5m6(float *dst, const uint16_t *src,
                                              int count) {
  using namespace deepseek::pto0571;
  (void)count;
  VecTile<uint16_t> input;
  VecTile<float> output;
  load(input, src);
  pto::TCVT(output, input);
  store(dst, output);
}

extern "C" void
deepseek_quant_swiglu_forward_per_token_i8(int8_t *dst, float *scales,
                                           const float *gate, const float *up,
                                           int tokens, int hidden) {
  using namespace deepseek::pto0571;
  (void)tokens;
  (void)hidden;
  VecTile<float> gate_tile;
  VecTile<float> up_tile;
  VecTile<float> activated;
  VecTile<float> scale;
  VecTile<int8_t> output;
  load(gate_tile, gate);
  load(up_tile, up);
  swiglu(activated, gate_tile, up_tile);
  quantize_rows(output, scale, activated,
                [&](VecTile<float> &ready) { store(scales, ready, 1); });
  store(dst, output);
}

extern "C" void deepseek_quant_swiglu_forward_per_channel_transpose_i8(
    int8_t *dst, float *scales, const float *gate, const float *up, int tokens,
    int hidden) {
  using namespace deepseek::pto0571;
  (void)tokens;
  (void)hidden;
  VecTile<float> gate_tile;
  VecTile<float> up_tile;
  VecTile<float> activated;
  VecTile<float> transposed;
  VecTile<float> scale;
  VecTile<int8_t> output;
  load(gate_tile, gate);
  load(up_tile, up);
  swiglu(activated, gate_tile, up_tile);
  pto::TTRANS(transposed, activated);
  quantize_rows(output, scale, transposed,
                [&](VecTile<float> &ready) { store(scales, ready, 1); });
  store(dst, output);
}

extern "C" void deepseek_quant_swiglu_backward_per_token_i8(
    int8_t *dgate, int8_t *dup, float *gate_scales, float *up_scales,
    const float *grad, const float *gate, const float *up, int tokens,
    int hidden) {
  using namespace deepseek::pto0571;
  (void)tokens;
  (void)hidden;
  VecTile<float> grad_tile;
  VecTile<float> gate_tile;
  VecTile<float> up_tile;
  VecTile<float> probability;
  VecTile<float> one;
  VecTile<float> one_minus_probability;
  VecTile<float> gate_factor;
  VecTile<float> derivative;
  VecTile<float> grad_gate_f32;
  VecTile<float> grad_up_f32;
  VecTile<float> gate_scale;
  VecTile<float> up_scale;
  VecTile<int8_t> grad_gate_i8;
  VecTile<int8_t> grad_up_i8;
  load(grad_tile, grad);
  load(gate_tile, gate);
  load(up_tile, up);
  sigmoid(probability, gate_tile);
  pto::TEXPANDS(one, 1.0f);
  pto::TSUB(one_minus_probability, one, probability);
  pto::TMUL(gate_factor, gate_tile, one_minus_probability);
  pto::TADDS(derivative, gate_factor, 1.0f);
  pto::TMUL(grad_gate_f32, grad_tile, up_tile);
  pto::TMUL(grad_gate_f32, grad_gate_f32, probability);
  pto::TMUL(grad_gate_f32, grad_gate_f32, derivative);
  pto::TMUL(grad_up_f32, grad_tile, gate_tile);
  pto::TMUL(grad_up_f32, grad_up_f32, probability);
  quantize_rows(grad_gate_i8, gate_scale, grad_gate_f32,
                [&](VecTile<float> &ready) { store(gate_scales, ready, 1); });
  quantize_rows(grad_up_i8, up_scale, grad_up_f32,
                [&](VecTile<float> &ready) { store(up_scales, ready, 1); });
  store(dgate, grad_gate_i8);
  store(dup, grad_up_i8);
}
