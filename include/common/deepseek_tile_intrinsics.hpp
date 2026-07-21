#ifndef PTO_COMMON_DEEPSEEK_TILE_INTRINSICS_HPP
#define PTO_COMMON_DEEPSEEK_TILE_INTRINSICS_HPP

#include <common/pto_tileop.hpp>

#include <stdint.h>

namespace deepseek::pto57 {

constexpr int kRows = 32;
constexpr int kCols = 32;
constexpr int kElements = kRows * kCols;

template <typename T>
using VecTile = pto::Tile<pto::Location::Vec, T, kRows, kCols,
                          pto::BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC>;

inline __attribute__((always_inline)) int tile_extent(int remaining,
                                                      int capacity) {
  return remaining < capacity ? remaining : capacity;
}

template <typename Fn>
inline __attribute__((always_inline)) void for_each_tile_2d(int rows, int cols,
                                                            Fn &&fn) {
  for (int row = 0; row < rows; row += kRows) {
    const int valid_rows = tile_extent(rows - row, kRows);
    for (int col = 0; col < cols; col += kCols) {
      const int valid_cols = tile_extent(cols - col, kCols);
      fn(row, col, valid_rows, valid_cols);
    }
  }
}

template <typename Fn>
inline __attribute__((always_inline)) void for_each_tile_1d(int count,
                                                            Fn &&fn) {
  for (int offset = 0; offset < count; offset += kCols)
    fn(offset, tile_extent(count - offset, kCols));
}

template <typename Fn>
inline __attribute__((always_inline)) void for_each_index(int count, Fn &&fn) {
  for (int index = 0; index < count; ++index)
    fn(index);
}

template <typename T>
inline __attribute__((always_inline)) void load(VecTile<T> &tile, const T *src,
                                                int valid_rows, int valid_cols,
                                                int row_stride) {
  tile.SetValidShape(valid_rows, valid_cols);
  tile.raw() =
      pto::linx::detail::tileTLoad<pto::detail::tileSizeCode<VecTile<T>>(),
                                   pto::detail::tileDTypeCode<VecTile<T>>(),
                                   pto::detail::kLayoutNorm>(
          src, static_cast<unsigned>(valid_cols),
          static_cast<unsigned>(valid_rows), kCols,
          static_cast<uint64_t>(row_stride) * sizeof(T));
}

template <typename T>
inline __attribute__((always_inline)) void store(T *dst, VecTile<T> &tile,
                                                 int row_stride) {
  pto::linx::detail::tileTStore<pto::detail::tileSizeCode<VecTile<T>>(),
                                pto::detail::tileDTypeCode<VecTile<T>>(),
                                pto::detail::kLayoutNorm>(
      dst, tile.raw(), static_cast<unsigned>(tile.GetValidCol()),
      static_cast<unsigned>(tile.GetValidRow()), kCols,
      static_cast<uint64_t>(row_stride) * sizeof(T));
}

template <typename T>
inline __attribute__((always_inline)) void load(VecTile<T> &tile,
                                                const T *src) {
  load(tile, src, kRows, kCols, kCols);
}

template <typename T>
inline __attribute__((always_inline)) void store(T *dst, VecTile<T> &tile) {
  store(dst, tile, kCols);
}

template <typename DstT, typename SrcT, typename Fn>
inline __attribute__((always_inline)) void
tilewise_unary(DstT *dst, const SrcT *src, int rows, int cols, Fn &&fn) {
  for_each_tile_2d(
      rows, cols, [&](int row, int col, int valid_rows, int valid_cols) {
        VecTile<SrcT> input(valid_rows, valid_cols);
        VecTile<DstT> output;
        load(input, src + row * cols + col, valid_rows, valid_cols, cols);
        fn(output, input);
        store(dst + row * cols + col, output, cols);
      });
}

template <typename DstT, typename LhsT, typename RhsT, typename Fn>
inline __attribute__((always_inline)) void
tilewise_binary(DstT *dst, const LhsT *lhs, const RhsT *rhs, int rows, int cols,
                Fn &&fn) {
  for_each_tile_2d(
      rows, cols, [&](int row, int col, int valid_rows, int valid_cols) {
        VecTile<LhsT> left(valid_rows, valid_cols);
        VecTile<RhsT> right(valid_rows, valid_cols);
        VecTile<DstT> output;
        load(left, lhs + row * cols + col, valid_rows, valid_cols, cols);
        load(right, rhs + row * cols + col, valid_rows, valid_cols, cols);
        fn(output, left, right);
        store(dst + row * cols + col, output, cols);
      });
}

inline __attribute__((always_inline)) void row_normalize(VecTile<float> &dst,
                                                         VecTile<float> &src) {
  VecTile<float> sum;
  VecTile<float> expanded(src.GetValidRow(), src.GetValidCol());
  pto::TROWSUM(sum, src);
  pto::TROWEXPAND(expanded, sum);
  pto::TDIV(dst, src, expanded);
}

inline __attribute__((always_inline)) void
rms_normalize(VecTile<float> &dst, VecTile<float> &src, float epsilon) {
  VecTile<float> square;
  VecTile<float> sum;
  VecTile<float> mean;
  VecTile<float> stabilized;
  VecTile<float> inverse;
  VecTile<float> expanded(src.GetValidRow(), src.GetValidCol());
  pto::TMUL(square, src, src);
  pto::TROWSUM(sum, square);
  pto::TDIVS(mean, sum, static_cast<float>(src.GetValidCol()));
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

inline __attribute__((always_inline)) void
swiglu(VecTile<float> &dst, VecTile<float> &gate, VecTile<float> &up) {
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
  VecTile<float> expanded(src.GetValidRow(), src.GetValidCol());
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
