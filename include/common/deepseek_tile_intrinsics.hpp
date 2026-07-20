#ifndef PTO_COMMON_DEEPSEEK_TILE_INTRINSICS_HPP
#define PTO_COMMON_DEEPSEEK_TILE_INTRINSICS_HPP

#include <common/pto_tileop.hpp>

#include <stdint.h>

namespace deepseek::pto57 {

constexpr int kRows = 32;
constexpr int kCols = 32;
constexpr int kElements = kRows * kCols;

template <typename T>
using Tensor = pto::global_tensor<T, pto::RowMajor<kRows, kCols>>;

template <typename T>
using VecTile =
    pto::Tile<pto::Location::Vec, T, kRows, kCols, pto::BLayout::RowMajor>;

template <typename T>
inline __attribute__((always_inline)) void load(VecTile<T> &tile,
                                                const T *src) {
  pto::global_iterator<Tensor<T>, VecTile<T>> iterator(const_cast<T *>(src));
  pto::TLOAD(tile, iterator(0, 0));
}

template <typename T>
inline __attribute__((always_inline)) void store(T *dst, VecTile<T> &tile) {
  pto::global_iterator<Tensor<T>, VecTile<T>> iterator(dst);
  pto::TSTORE(iterator(0, 0), tile);
}

inline __attribute__((always_inline)) void row_normalize(VecTile<float> &dst,
                                                         VecTile<float> &src) {
  VecTile<float> sum;
  VecTile<float> expanded;
  pto::TROWSUM(sum, src);
  pto::TROWEXPAND(expanded, sum);
  pto::TDIV(dst, src, expanded);
}

inline __attribute__((always_inline)) void rms_normalize(VecTile<float> &dst,
                                                         VecTile<float> &src,
                                                         float epsilon) {
  VecTile<float> square;
  VecTile<float> sum;
  VecTile<float> mean;
  VecTile<float> stabilized;
  VecTile<float> inverse;
  VecTile<float> expanded;
  pto::TMUL(square, src, src);
  pto::TROWSUM(sum, square);
  pto::TDIVS(mean, sum, static_cast<float>(kCols));
  pto::TADDS(stabilized, mean, epsilon);
  pto::TRSQRT(inverse, stabilized);
  pto::TROWEXPAND(expanded, inverse);
  pto::TMUL(dst, src, expanded);
}

inline __attribute__((always_inline)) void sigmoid(VecTile<float> &dst,
                                                   VecTile<float> &src) {
  VecTile<float> negative;
  VecTile<float> exponent;
  VecTile<float> denominator;
  pto::TMULS(negative, src, -1.0f);
  pto::TEXP(exponent, negative);
  pto::TADDS(denominator, exponent, 1.0f);
  pto::TRECIP(dst, denominator);
}

inline __attribute__((always_inline)) void swiglu(VecTile<float> &dst,
                                                  VecTile<float> &gate,
                                                  VecTile<float> &up) {
  VecTile<float> probability;
  VecTile<float> activated;
  sigmoid(probability, gate);
  pto::TMUL(activated, gate, probability);
  pto::TMUL(dst, activated, up);
}

inline __attribute__((always_inline)) void quantize_rows(VecTile<int8_t> &dst,
                                                         VecTile<float> &scale,
                                                         VecTile<float> &src) {
  VecTile<float> absolute;
  VecTile<float> maximum;
  VecTile<float> expanded;
  VecTile<float> normalized;
  pto::TABS(absolute, src);
  pto::TROWMAX(maximum, absolute);
  pto::TDIVS(scale, maximum, 127.0f);
  pto::TROWEXPAND(expanded, scale);
  pto::TDIV(normalized, src, expanded);
  pto::TCVT(dst, normalized);
}

inline __attribute__((always_inline)) void sinkhorn_step(VecTile<float> &dst,
                                                         VecTile<float> &src) {
  VecTile<float> positive;
  VecTile<float> row_normalized;
  VecTile<float> transposed;
  VecTile<float> column_normalized_t;
  pto::TEXP(positive, src);
  row_normalize(row_normalized, positive);
  pto::TTRANSPOSE(transposed, row_normalized);
  row_normalize(column_normalized_t, transposed);
  pto::TTRANSPOSE(dst, column_normalized_t);
}

} // namespace deepseek::pto57

#endif // PTO_COMMON_DEEPSEEK_TILE_INTRINSICS_HPP
