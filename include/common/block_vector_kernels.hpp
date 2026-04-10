#ifndef PTO_COMMON_BLOCK_VECTOR_KERNELS_HPP
#define PTO_COMMON_BLOCK_VECTOR_KERNELS_HPP

#include <common/block_vector_compat.hpp>

namespace pto {
namespace kernels {

template <typename tileSrc, typename tileMax>
void __vec__ new_max_1src(typename tileMax::TileDType __out__ scale,
                          typename tileMax::TileDType __out__ new_max,
                          const typename tileSrc::TileDType __in__ src,
                          const typename tileMax::TileDType __in__ old_max,
                          const typename tileSrc::DType src_scale) {
  const size_t i = blkv::blkv_get_index_x();

  __vbuf__ typename tileSrc::DType *src_ptr = blkv::blkv_get_tile_ptr(src);
  __vbuf__ typename tileMax::DType *new_max_ptr =
      blkv::blkv_get_tile_ptr(new_max);
  __vbuf__ typename tileMax::DType *old_max_ptr =
      blkv::blkv_get_tile_ptr(old_max);
  __vbuf__ typename tileMax::DType *scale_ptr = blkv::blkv_get_tile_ptr(scale);

  const size_t max_idx = i * tileMax::RowStride;
  typename tileMax::DType upd_max = old_max_ptr[max_idx];

#pragma clang loop unroll(full)
  for (size_t j = 0; j < tileSrc::ValidCol; ++j) {
    const size_t src_idx = i * tileSrc::RowStride + j * tileSrc::ColStride;
    upd_max = blkv::blkv_max(
        upd_max, static_cast<typename tileMax::DType>(src_ptr[src_idx] * src_scale));
  }

  new_max_ptr[max_idx] = upd_max;
  scale_ptr[max_idx] = blkv::blkv_fexp(old_max_ptr[max_idx] - upd_max);
}

template <typename tileSrc, typename tileSrcCast, typename tileMax,
          typename tileSum, typename tileScale>
void __vec__ src_exp_1src_with_new_sum(
    typename tileSum::TileDType __out__ new_sum,
    typename tileSrcCast::TileDType __out__ src_exp,
    const typename tileSrc::TileDType __in__ src,
    const typename tileMax::TileDType __in__ new_max,
    const typename tileSum::TileDType __in__ old_sum,
    const typename tileScale::TileDType __in__ scale,
    const typename tileSrc::DType src_scale) {
  const size_t i = blkv::blkv_get_index_x();
  const size_t idx_max = i * tileMax::RowStride;
  const size_t idx_sum = i * tileSum::RowStride;

  __vbuf__ typename tileSum::DType *old_sum_ptr = blkv::blkv_get_tile_ptr(old_sum);
  __vbuf__ typename tileScale::DType *scale_ptr = blkv::blkv_get_tile_ptr(scale);
  __vbuf__ typename tileSrc::DType *src_ptr = blkv::blkv_get_tile_ptr(src);
  __vbuf__ typename tileSrcCast::DType *src_exp_ptr =
      blkv::blkv_get_tile_ptr(src_exp);
  __vbuf__ typename tileSum::DType *new_sum_ptr = blkv::blkv_get_tile_ptr(new_sum);
  __vbuf__ typename tileMax::DType *new_max_ptr = blkv::blkv_get_tile_ptr(new_max);

  typename tileSum::DType upd_sum = old_sum_ptr[idx_sum] * scale_ptr[idx_sum];
  const typename tileMax::DType new_max_val = new_max_ptr[idx_max];

#pragma clang loop unroll(full)
  for (size_t j = 0; j < tileSrc::ValidCol; ++j) {
    const size_t idx = i * tileSrc::RowStride + j * tileSrc::ColStride;
    const typename tileSrc::DType exp_val =
        blkv::blkv_fexp(src_ptr[idx] * src_scale - new_max_val);
    src_exp_ptr[idx] = static_cast<typename tileSrcCast::DType>(exp_val);
    upd_sum += static_cast<typename tileSum::DType>(exp_val);
  }

  new_sum_ptr[idx_sum] = upd_sum;
}

template <typename tileSrc, typename tileMax>
void __vpar__ new_max_2src(typename tileMax::TileDType __out__ scale,
                           typename tileMax::TileDType __out__ new_max,
                           const typename tileSrc::TileDType __in__ src0,
                           const typename tileSrc::TileDType __in__ src1,
                           const typename tileMax::TileDType __in__ old_max,
                           const typename tileSrc::DType src_scale) {
  const size_t i = blkv::blkv_get_index_x();

  __vbuf__ typename tileSrc::DType *src0_ptr = blkv::blkv_get_tile_ptr(src0);
  __vbuf__ typename tileSrc::DType *src1_ptr = blkv::blkv_get_tile_ptr(src1);
  __vbuf__ typename tileMax::DType *new_max_ptr =
      blkv::blkv_get_tile_ptr(new_max);
  __vbuf__ typename tileMax::DType *old_max_ptr =
      blkv::blkv_get_tile_ptr(old_max);
  __vbuf__ typename tileMax::DType *scale_ptr = blkv::blkv_get_tile_ptr(scale);

  const size_t max_idx = i * tileMax::RowStride;
  typename tileMax::DType upd_max = old_max_ptr[max_idx];

#pragma clang loop unroll(full)
  for (size_t j = 0; j < tileSrc::ValidCol; ++j) {
    const size_t src_idx = i * tileSrc::RowStride + j * tileSrc::ColStride;
    upd_max = blkv::blkv_max(
        upd_max,
        static_cast<typename tileMax::DType>(src0_ptr[src_idx] * src_scale));
    upd_max = blkv::blkv_max(
        upd_max,
        static_cast<typename tileMax::DType>(src1_ptr[src_idx] * src_scale));
  }

  new_max_ptr[max_idx] = upd_max;
  scale_ptr[max_idx] = blkv::blkv_fexp(old_max_ptr[max_idx] - upd_max);
}

template <typename tileSrc, typename tileSrcCast, typename tileMax,
          typename tileSum, typename tileScale>
void __vpar__ src_exp_2src_with_new_sum(
    typename tileSum::TileDType __out__ new_sum,
    typename tileSrcCast::TileDType __out__ src_exp0,
    typename tileSrcCast::TileDType __out__ src_exp1,
    const typename tileSrc::TileDType __in__ src0,
    const typename tileSrc::TileDType __in__ src1,
    const typename tileMax::TileDType __in__ new_max,
    const typename tileSum::TileDType __in__ old_sum,
    const typename tileScale::TileDType __in__ scale,
    const typename tileSrc::DType src_scale) {
  const size_t i = blkv::blkv_get_index_x();
  const size_t idx_max = i * tileMax::RowStride;
  const size_t idx_sum = i * tileSum::RowStride;

  __vbuf__ typename tileSum::DType *old_sum_ptr = blkv::blkv_get_tile_ptr(old_sum);
  __vbuf__ typename tileScale::DType *scale_ptr = blkv::blkv_get_tile_ptr(scale);
  __vbuf__ typename tileSrc::DType *src0_ptr = blkv::blkv_get_tile_ptr(src0);
  __vbuf__ typename tileSrc::DType *src1_ptr = blkv::blkv_get_tile_ptr(src1);
  __vbuf__ typename tileSrcCast::DType *src_exp0_ptr =
      blkv::blkv_get_tile_ptr(src_exp0);
  __vbuf__ typename tileSrcCast::DType *src_exp1_ptr =
      blkv::blkv_get_tile_ptr(src_exp1);
  __vbuf__ typename tileSum::DType *new_sum_ptr = blkv::blkv_get_tile_ptr(new_sum);
  __vbuf__ typename tileMax::DType *new_max_ptr = blkv::blkv_get_tile_ptr(new_max);

  typename tileSum::DType upd_sum = old_sum_ptr[idx_sum] * scale_ptr[idx_sum];
  const typename tileMax::DType new_max_val = new_max_ptr[idx_max];

#pragma clang loop unroll(full)
  for (size_t j = 0; j < tileSrc::ValidCol; ++j) {
    const size_t idx = i * tileSrc::RowStride + j * tileSrc::ColStride;
    const typename tileSrc::DType exp0 =
        blkv::blkv_fexp(src0_ptr[idx] * src_scale - new_max_val);
    const typename tileSrc::DType exp1 =
        blkv::blkv_fexp(src1_ptr[idx] * src_scale - new_max_val);
    src_exp0_ptr[idx] = static_cast<typename tileSrcCast::DType>(exp0);
    src_exp1_ptr[idx] = static_cast<typename tileSrcCast::DType>(exp1);
    upd_sum += static_cast<typename tileSum::DType>(exp0 + exp1);
  }

  new_sum_ptr[idx_sum] = upd_sum;
}

template <typename tileSrc, typename tileMax>
void __vpar__ new_max_4src(typename tileMax::TileDType __out__ scale,
                           typename tileMax::TileDType __out__ new_max,
                           const typename tileSrc::TileDType __in__ src0,
                           const typename tileSrc::TileDType __in__ src1,
                           const typename tileSrc::TileDType __in__ src2,
                           const typename tileSrc::TileDType __in__ src3,
                           const typename tileMax::TileDType __in__ old_max,
                           const typename tileSrc::DType src_scale) {
  const size_t i = blkv::blkv_get_index_x();

  __vbuf__ typename tileSrc::DType *src0_ptr = blkv::blkv_get_tile_ptr(src0);
  __vbuf__ typename tileSrc::DType *src1_ptr = blkv::blkv_get_tile_ptr(src1);
  __vbuf__ typename tileSrc::DType *src2_ptr = blkv::blkv_get_tile_ptr(src2);
  __vbuf__ typename tileSrc::DType *src3_ptr = blkv::blkv_get_tile_ptr(src3);
  __vbuf__ typename tileMax::DType *new_max_ptr =
      blkv::blkv_get_tile_ptr(new_max);
  __vbuf__ typename tileMax::DType *old_max_ptr =
      blkv::blkv_get_tile_ptr(old_max);
  __vbuf__ typename tileMax::DType *scale_ptr = blkv::blkv_get_tile_ptr(scale);

  const size_t max_idx = i * tileMax::RowStride;
  typename tileMax::DType upd_max = old_max_ptr[max_idx];

#pragma clang loop unroll(full)
  for (size_t j = 0; j < tileSrc::ValidCol; ++j) {
    const size_t src_idx = i * tileSrc::RowStride + j * tileSrc::ColStride;
    upd_max = blkv::blkv_max(
        upd_max,
        static_cast<typename tileMax::DType>(src0_ptr[src_idx] * src_scale));
    upd_max = blkv::blkv_max(
        upd_max,
        static_cast<typename tileMax::DType>(src1_ptr[src_idx] * src_scale));
    upd_max = blkv::blkv_max(
        upd_max,
        static_cast<typename tileMax::DType>(src2_ptr[src_idx] * src_scale));
    upd_max = blkv::blkv_max(
        upd_max,
        static_cast<typename tileMax::DType>(src3_ptr[src_idx] * src_scale));
  }

  new_max_ptr[max_idx] = upd_max;
  scale_ptr[max_idx] = blkv::blkv_fexp(old_max_ptr[max_idx] - upd_max);
}

template <typename tileSrc, typename tileSrcCast, typename tileMax,
          typename tileSum>
void __vpar__ src_exp_2src_with_local_sum(
    typename tileSum::TileDType __out__ local_sum,
    typename tileSrcCast::TileDType __out__ src_exp0,
    typename tileSrcCast::TileDType __out__ src_exp1,
    const typename tileSrc::TileDType __in__ src0,
    const typename tileSrc::TileDType __in__ src1,
    const typename tileMax::TileDType __in__ new_max,
    const typename tileSrc::DType src_scale) {
  const size_t i = blkv::blkv_get_index_x();
  const size_t idx_max = i * tileMax::RowStride;
  const size_t idx_sum = i * tileSum::RowStride;

  __vbuf__ typename tileSrc::DType *src0_ptr = blkv::blkv_get_tile_ptr(src0);
  __vbuf__ typename tileSrc::DType *src1_ptr = blkv::blkv_get_tile_ptr(src1);
  __vbuf__ typename tileSrcCast::DType *src_exp0_ptr =
      blkv::blkv_get_tile_ptr(src_exp0);
  __vbuf__ typename tileSrcCast::DType *src_exp1_ptr =
      blkv::blkv_get_tile_ptr(src_exp1);
  __vbuf__ typename tileMax::DType *new_max_ptr = blkv::blkv_get_tile_ptr(new_max);
  __vbuf__ typename tileSum::DType *local_sum_ptr =
      blkv::blkv_get_tile_ptr(local_sum);

  const typename tileMax::DType new_max_val = new_max_ptr[idx_max];
  typename tileSum::DType upd_sum = 0;

#pragma clang loop unroll(full)
  for (size_t j = 0; j < tileSrc::ValidCol; ++j) {
    const size_t idx = i * tileSrc::RowStride + j * tileSrc::ColStride;
    const typename tileSrc::DType exp0 =
        blkv::blkv_fexp(src0_ptr[idx] * src_scale - new_max_val);
    const typename tileSrc::DType exp1 =
        blkv::blkv_fexp(src1_ptr[idx] * src_scale - new_max_val);
    src_exp0_ptr[idx] = static_cast<typename tileSrcCast::DType>(exp0);
    src_exp1_ptr[idx] = static_cast<typename tileSrcCast::DType>(exp1);
    upd_sum += static_cast<typename tileSum::DType>(exp0 + exp1);
  }

  local_sum_ptr[idx_sum] = upd_sum;
}

template <typename tileScale, typename tileSum>
void __vpar__ new_sum_of_2_loc_sum(
    typename tileSum::TileDType __out__ new_sum,
    const typename tileSum::TileDType __in__ local_sum_0,
    const typename tileSum::TileDType __in__ local_sum_1,
    const typename tileSum::TileDType __in__ old_sum,
    const typename tileScale::TileDType __in__ scale) {
  const size_t i = blkv::blkv_get_index_x();
  const size_t sum_idx = i * tileSum::RowStride;

  __vbuf__ typename tileSum::DType *new_sum_ptr = blkv::blkv_get_tile_ptr(new_sum);
  __vbuf__ typename tileSum::DType *local_sum_0_ptr =
      blkv::blkv_get_tile_ptr(local_sum_0);
  __vbuf__ typename tileSum::DType *local_sum_1_ptr =
      blkv::blkv_get_tile_ptr(local_sum_1);
  __vbuf__ typename tileSum::DType *old_sum_ptr = blkv::blkv_get_tile_ptr(old_sum);
  __vbuf__ typename tileScale::DType *scale_ptr = blkv::blkv_get_tile_ptr(scale);

  new_sum_ptr[sum_idx] = old_sum_ptr[sum_idx] * scale_ptr[sum_idx] +
                         local_sum_0_ptr[sum_idx] + local_sum_1_ptr[sum_idx];
}

template <typename tileOut, typename tileScale>
void __vpar__ global_update(typename tileOut::TileDType __out__ dst,
                            const typename tileOut::TileDType __in__ old_val,
                            const typename tileOut::TileDType __in__ new_val,
                            const typename tileScale::TileDType __in__ scale) {
  const size_t i = blkv::blkv_get_index_x();
  const size_t j = blkv::blkv_get_index_y();
  const size_t idx = i * tileOut::RowStride + j * tileOut::ColStride;
  const size_t scale_idx = i * tileScale::RowStride;

  __vbuf__ typename tileOut::DType *dst_ptr = blkv::blkv_get_tile_ptr(dst);
  __vbuf__ typename tileOut::DType *old_ptr = blkv::blkv_get_tile_ptr(old_val);
  __vbuf__ typename tileOut::DType *new_ptr = blkv::blkv_get_tile_ptr(new_val);
  __vbuf__ typename tileScale::DType *scale_ptr = blkv::blkv_get_tile_ptr(scale);

  dst_ptr[idx] = old_ptr[idx] * scale_ptr[scale_idx] + new_ptr[idx];
}

template <typename tileOut, typename tileSum>
void __vpar__ normalize_with_sum(typename tileOut::TileDType __out__ dst,
                                 const typename tileOut::TileDType __in__ src,
                                 const typename tileSum::TileDType __in__ sum) {
  const size_t i = blkv::blkv_get_index_x();
  const size_t j = blkv::blkv_get_index_y();
  const size_t idx = i * tileOut::RowStride + j * tileOut::ColStride;
  const size_t sum_idx = i * tileSum::RowStride;

  __vbuf__ typename tileOut::DType *dst_ptr = blkv::blkv_get_tile_ptr(dst);
  __vbuf__ typename tileOut::DType *src_ptr = blkv::blkv_get_tile_ptr(src);
  __vbuf__ typename tileSum::DType *sum_ptr = blkv::blkv_get_tile_ptr(sum);

  const typename tileSum::DType denom = sum_ptr[sum_idx];
  dst_ptr[idx] = denom == 0 ? static_cast<typename tileOut::DType>(0)
                            : static_cast<typename tileOut::DType>(src_ptr[idx] / denom);
}

template <typename tileSrc>
void __vec__ mask_attention_tile(typename tileSrc::TileDType __out__ src,
                                 int q_base, int k_base, int seq_len,
                                 bool causal, int window) {
  const int i = static_cast<int>(blkv::blkv_get_index_x());
  const int j = static_cast<int>(blkv::blkv_get_index_y());
  const size_t idx = static_cast<size_t>(i) * tileSrc::RowStride +
                     static_cast<size_t>(j) * tileSrc::ColStride;

  __vbuf__ typename tileSrc::DType *src_ptr = blkv::blkv_get_tile_ptr(src);
  const int q = q_base + i;
  const int k = k_base + j;
  bool masked = q >= seq_len || k >= seq_len;
  if (!masked && causal && k > q)
    masked = true;
  if (!masked && window >= 0) {
    const int delta = q > k ? (q - k) : (k - q);
    masked = delta > window;
  }
  if (masked)
    src_ptr[idx] = static_cast<typename tileSrc::DType>(-1e30f);
}

template <typename tileSrc, typename tileSum>
void __vec__ row_square_sum(typename tileSum::TileDType __out__ sum,
                            const typename tileSrc::TileDType __in__ src) {
  const size_t i = blkv::blkv_get_index_x();
  const size_t sum_idx = i * tileSum::RowStride;

  __vbuf__ typename tileSrc::DType *src_ptr = blkv::blkv_get_tile_ptr(src);
  __vbuf__ typename tileSum::DType *sum_ptr = blkv::blkv_get_tile_ptr(sum);

  typename tileSum::DType acc = 0;
#pragma clang loop unroll(full)
  for (size_t j = 0; j < tileSrc::ValidCol; ++j) {
    const size_t idx = i * tileSrc::RowStride + j * tileSrc::ColStride;
    const typename tileSrc::DType val = src_ptr[idx];
    acc += static_cast<typename tileSum::DType>(val * val);
  }
  sum_ptr[sum_idx] = acc;
}

template <typename tileOut, typename tileSrc, typename tileGamma,
          typename tileScale>
void __vec__ rmsnorm_apply(typename tileOut::TileDType __out__ out,
                           const typename tileSrc::TileDType __in__ src,
                           const typename tileGamma::TileDType __in__ gamma,
                           const typename tileScale::TileDType __in__ inv_rms) {
  const size_t i = blkv::blkv_get_index_x();
  const size_t j = blkv::blkv_get_index_y();
  const size_t out_idx = i * tileOut::RowStride + j * tileOut::ColStride;
  const size_t src_idx = i * tileSrc::RowStride + j * tileSrc::ColStride;
  const size_t gamma_idx = j * tileGamma::ColStride;
  const size_t inv_idx = i * tileScale::RowStride;

  __vbuf__ typename tileOut::DType *out_ptr = blkv::blkv_get_tile_ptr(out);
  __vbuf__ typename tileSrc::DType *src_ptr = blkv::blkv_get_tile_ptr(src);
  __vbuf__ typename tileGamma::DType *gamma_ptr = blkv::blkv_get_tile_ptr(gamma);
  __vbuf__ typename tileScale::DType *inv_ptr = blkv::blkv_get_tile_ptr(inv_rms);

  out_ptr[out_idx] = static_cast<typename tileOut::DType>(
      src_ptr[src_idx] * gamma_ptr[gamma_idx] * inv_ptr[inv_idx]);
}

template <int S, int D, int VD, int TM, int TK, int YDim = 2,
          bool Causal = false>
inline void mixed_attention_f32(float *out_ptr, float *q_ptr, float *k_ptr,
                                float *v_ptr, int window = -1) {
  static_assert(S > 0 && D > 0 && VD > 0 && TM > 0 && TK > 0,
                "shape params must be positive");
  static_assert(S % TM == 0 && S % TK == 0, "S must be divisible by TM/TK");
  static_assert(YDim == 1 || YDim == 2 || YDim == 4,
                "mixed_attention_f32 currently supports YDim=1, YDim=2, or YDim=4");

  using gmQ = global_tensor<float, RowMajor<S, D>>;
  using gmK = global_tensor<float, ColMajor<D, S>>;
  using gmV = global_tensor<float, RowMajor<S, VD>>;
  using gmO = global_tensor<float, RowMajor<S, VD>>;

  using tileQ = TileLeft<float, TM, D>;
  using tileK = TileRight<float, D, TK>;
  using tileV = TileRight<float, TK, VD>;
  using tileWOut = TileAcc<float, TM, TK>;
  using tileW = Tile<Location::Vec, float, TM, TK, BLayout::RowMajor>;
  using tileWExp = Tile<Location::Vec, float, TM, TK, BLayout::RowMajor>;
  using tileWLeft = TileLeft<float, TM, TK>;
  using tileOutAcc = TileAcc<float, TM, VD>;
  using tileOut = Tile<Location::Vec, float, TM, VD, BLayout::RowMajor>;
  using tileMax = Tile<Location::Vec, float, TM, 1, BLayout::RowMajor>;
  using tileSum = Tile<Location::Vec, float, TM, 1, BLayout::RowMajor>;
  using tileScale = Tile<Location::Vec, float, TM, 1, BLayout::RowMajor>;

  using itQ = global_iterator<gmQ, tileQ>;
  using itK = global_iterator<gmK, tileK>;
  using itV = global_iterator<gmV, tileV>;
  using itO = global_iterator<gmO, tileOut>;

  itQ gQ(q_ptr);
  itK gK(k_ptr);
  itV gV(v_ptr);
  itO gO(out_ptr);

  constexpr int kQTiles = S / TM;
  constexpr int kKTiles = S / TK;
  const float scale = 1.0f / blkv::blkv_fsqrt(static_cast<float>(D));

  for (int qi = 0; qi < kQTiles; ++qi) {
    tileQ tQ;
    TLOAD(tQ, gQ(qi, 0));

    tileMax tMax;
    tileSum tSum(0.0f);
    tileOut tOut(0.0f);
    TEXPANDS(tMax, -1e30f);
    bool have_output = false;

    for (int kj = 0; kj < kKTiles; kj += YDim) {
      const int active_tiles = (kj + YDim <= kKTiles) ? YDim : (kKTiles - kj);

      tileK tK[YDim];
      tileV tV[YDim];
      tileW tW[YDim];
      tileWExp tExpW[YDim];

      for (int y = 0; y < active_tiles; ++y) {
        tileWOut tWOut;
        TLOAD(tK[y], gK(0, kj + y));
        TLOAD(tV[y], gV(kj + y, 0));
        TMATMUL(tWOut, tQ, tK[y]);
        TCVT(tW[y], tWOut);
        TMULS(tW[y], tW[y], scale);
        if (Causal || window >= 0) {
          blkv::blkv_for_2d(tileW::ValidRow, tileW::ValidCol, [&] {
            mask_attention_tile<tileW>(tW[y].data(), qi * TM, (kj + y) * TK, S,
                                       Causal, window);
          });
        }
      }

      tileScale tScale;
      tileMax tNewMax;
      tileSum tNewSum;
      auto generic_softmax_update = [&] {
        blkv::blkv_for_1d(tileMax::ValidRow, [&] {
          const size_t i = blkv::blkv_get_index_x();
          __vbuf__ float *new_max_ptr = blkv::blkv_get_tile_ptr(tNewMax.data());
          __vbuf__ float *old_max_ptr = blkv::blkv_get_tile_ptr(tMax.data());
          __vbuf__ float *scale_ptr = blkv::blkv_get_tile_ptr(tScale.data());
          float upd_max = old_max_ptr[i * tileMax::RowStride];

          for (int y = 0; y < active_tiles; ++y) {
            __vbuf__ float *src_ptr = blkv::blkv_get_tile_ptr(tW[y].data());
            for (size_t j = 0; j < tileW::ValidCol; ++j) {
              const size_t idx =
                  i * tileW::RowStride + j * tileW::ColStride;
              upd_max = blkv::blkv_max(upd_max, src_ptr[idx]);
            }
          }

          const size_t max_idx = i * tileMax::RowStride;
          new_max_ptr[max_idx] = upd_max;
          scale_ptr[max_idx] = blkv::blkv_fexp(old_max_ptr[max_idx] - upd_max);
        });

        blkv::blkv_for_1d(tileSum::ValidRow, [&] {
          const size_t i = blkv::blkv_get_index_x();
          __vbuf__ float *new_sum_ptr = blkv::blkv_get_tile_ptr(tNewSum.data());
          __vbuf__ float *old_sum_ptr = blkv::blkv_get_tile_ptr(tSum.data());
          __vbuf__ float *scale_ptr = blkv::blkv_get_tile_ptr(tScale.data());
          __vbuf__ float *new_max_ptr = blkv::blkv_get_tile_ptr(tNewMax.data());
          float upd_sum = old_sum_ptr[i * tileSum::RowStride] *
                          scale_ptr[i * tileScale::RowStride];
          const float new_max_val = new_max_ptr[i * tileMax::RowStride];

          for (int y = 0; y < active_tiles; ++y) {
            __vbuf__ float *src_ptr = blkv::blkv_get_tile_ptr(tW[y].data());
            __vbuf__ float *exp_ptr = blkv::blkv_get_tile_ptr(tExpW[y].data());
            for (size_t j = 0; j < tileW::ValidCol; ++j) {
              const size_t idx =
                  i * tileW::RowStride + j * tileW::ColStride;
              const float exp_val = blkv::blkv_fexp(src_ptr[idx] - new_max_val);
              exp_ptr[idx] = exp_val;
              upd_sum += exp_val;
            }
          }

          new_sum_ptr[i * tileSum::RowStride] = upd_sum;
        });
      };
#if defined(__LINXISA__)
      if constexpr (YDim == 2) {
        new_max_2src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
            tScale.data(), tNewMax.data(), tW[0].data(), tW[1].data(),
            tMax.data(), 1.0f);
        src_exp_2src_with_new_sum<tileW, tileWExp, tileMax, tileSum, tileScale>
            <<<tileSum::ValidRow, 1, 1>>>(
                tNewSum.data(), tExpW[0].data(), tExpW[1].data(),
                tW[0].data(), tW[1].data(), tNewMax.data(), tSum.data(),
                tScale.data(), 1.0f);
      } else if constexpr (YDim == 4) {
        if (active_tiles == 4) {
          tileSum tLocalSum[2];
          new_max_4src<tileW, tileMax><<<tileMax::ValidRow, 1, 1>>>(
              tScale.data(), tNewMax.data(), tW[0].data(), tW[1].data(),
              tW[2].data(), tW[3].data(), tMax.data(), 1.0f);
          src_exp_2src_with_local_sum<tileW, tileWExp, tileMax, tileSum>
              <<<tileSum::ValidRow, 1, 1>>>(
                  tLocalSum[0].data(), tExpW[0].data(), tExpW[1].data(),
                  tW[0].data(), tW[1].data(), tNewMax.data(), 1.0f);
          src_exp_2src_with_local_sum<tileW, tileWExp, tileMax, tileSum>
              <<<tileSum::ValidRow, 1, 1>>>(
                  tLocalSum[1].data(), tExpW[2].data(), tExpW[3].data(),
                  tW[2].data(), tW[3].data(), tNewMax.data(), 1.0f);
          new_sum_of_2_loc_sum<tileScale, tileSum>
              <<<tileSum::ValidRow, 1, 1>>>(
                  tNewSum.data(), tLocalSum[0].data(), tLocalSum[1].data(),
                  tSum.data(), tScale.data());
        } else {
          generic_softmax_update();
        }
      } else
#endif
      if (active_tiles == 1) {
        blkv::blkv_for_1d(tileMax::ValidRow, [&] {
          new_max_1src<tileW, tileMax>(tScale.data(), tNewMax.data(),
                                       tW[0].data(), tMax.data(), 1.0f);
        });
        blkv::blkv_for_1d(tileSum::ValidRow, [&] {
          src_exp_1src_with_new_sum<tileW, tileWExp, tileMax, tileSum, tileScale>(
              tNewSum.data(), tExpW[0].data(), tW[0].data(), tNewMax.data(),
              tSum.data(), tScale.data(), 1.0f);
        });
      } else if (active_tiles == 2) {
        blkv::blkv_for_1d(tileMax::ValidRow, [&] {
          new_max_2src<tileW, tileMax>(tScale.data(), tNewMax.data(),
                                       tW[0].data(), tW[1].data(),
                                       tMax.data(), 1.0f);
        });
        blkv::blkv_for_1d(tileSum::ValidRow, [&] {
          src_exp_2src_with_new_sum<tileW, tileWExp, tileMax, tileSum, tileScale>(
              tNewSum.data(), tExpW[0].data(), tExpW[1].data(), tW[0].data(),
              tW[1].data(), tNewMax.data(), tSum.data(), tScale.data(), 1.0f);
        });
      } else if (active_tiles == 4) {
        tileSum tLocalSum[2];
        blkv::blkv_for_1d(tileMax::ValidRow, [&] {
          new_max_4src<tileW, tileMax>(tScale.data(), tNewMax.data(),
                                       tW[0].data(), tW[1].data(),
                                       tW[2].data(), tW[3].data(),
                                       tMax.data(), 1.0f);
        });
        blkv::blkv_for_1d(tileSum::ValidRow, [&] {
          src_exp_2src_with_local_sum<tileW, tileWExp, tileMax, tileSum>(
              tLocalSum[0].data(), tExpW[0].data(), tExpW[1].data(),
              tW[0].data(), tW[1].data(), tNewMax.data(), 1.0f);
          src_exp_2src_with_local_sum<tileW, tileWExp, tileMax, tileSum>(
              tLocalSum[1].data(), tExpW[2].data(), tExpW[3].data(),
              tW[2].data(), tW[3].data(), tNewMax.data(), 1.0f);
          new_sum_of_2_loc_sum<tileScale, tileSum>(
              tNewSum.data(), tLocalSum[0].data(), tLocalSum[1].data(),
              tSum.data(), tScale.data());
        });
      } else {
        generic_softmax_update();
      }

      tileOutAcc tPVAcc;
      tileWLeft tWLeft;
      TCVT(tWLeft, tExpW[0]);
      TMATMUL(tPVAcc, tWLeft, tV[0]);
      for (int y = 1; y < active_tiles; ++y) {
        TCVT(tWLeft, tExpW[y]);
        MATMACC(tPVAcc, tWLeft, tV[y]);
      }

      tileOut tPV;
      TCVT(tPV, tPVAcc);
      if (!have_output) {
        tOut = tPV;
        have_output = true;
      } else {
#if defined(__LINXISA__)
        if constexpr (YDim == 2) {
          global_update<tileOut, tileScale>
              <<<tileOut::ValidRow, tileOut::ValidCol, 1>>>(
                  tOut.data(), tOut.data(), tPV.data(), tScale.data());
        } else
#endif
        blkv::blkv_for_2d(tileOut::ValidRow, tileOut::ValidCol, [&] {
          global_update<tileOut, tileScale>(tOut.data(), tOut.data(),
                                            tPV.data(), tScale.data());
        });
      }

      tMax = tNewMax;
      tSum = tNewSum;
    }

#if defined(__LINXISA__)
    if constexpr (YDim == 2) {
      normalize_with_sum<tileOut, tileSum>
          <<<tileOut::ValidRow, tileOut::ValidCol, 1>>>(
              tOut.data(), tOut.data(), tSum.data());
    } else
#endif
    blkv::blkv_for_2d(tileOut::ValidRow, tileOut::ValidCol, [&] {
      normalize_with_sum<tileOut, tileSum>(tOut.data(), tOut.data(),
                                           tSum.data());
    });
    TSTORE(gO(qi, 0), tOut);
  }
}

template <int Tokens, int D>
inline void mixed_rmsnorm_f32(float *out_ptr, float *x_ptr, float *gamma_ptr,
                              float eps) {
  static_assert(Tokens > 0 && D > 0, "shape params must be positive");

  using gmX = global_tensor<float, RowMajor<Tokens, D>>;
  using gmG = global_tensor<float, RowMajor<1, D>>;
  using gmO = global_tensor<float, RowMajor<Tokens, D>>;

  using tileVec = Tile<Location::Vec, float, 1, D, BLayout::RowMajor>;
  using tileSum = Tile<Location::Vec, float, 1, 1, BLayout::RowMajor>;

  using itX = global_iterator<gmX, tileVec>;
  using itG = global_iterator<gmG, tileVec>;
  using itO = global_iterator<gmO, tileVec>;

  itX gX(x_ptr);
  itG gG(gamma_ptr);
  itO gO(out_ptr);

  tileVec tGamma;
  TLOAD(tGamma, gG(0, 0));

  for (int token = 0; token < Tokens; ++token) {
    tileVec tX;
    tileVec tOut;
    tileSum tSqSum;
    tileSum tInv;
    TLOAD(tX, gX(token, 0));
    blkv::blkv_for_1d(tileSum::ValidRow, [&] {
      row_square_sum<tileVec, tileSum>(tSqSum.data(), tX.data());
    });

    __vbuf__ float *inv_ptr = blkv::blkv_get_tile_ptr(tInv.data());
    __vbuf__ float *sum_ptr = blkv::blkv_get_tile_ptr(tSqSum.data());
    inv_ptr[0] =
        1.0f / blkv::blkv_fsqrt(sum_ptr[0] / static_cast<float>(D) + eps);

    blkv::blkv_for_2d(tileVec::ValidRow, tileVec::ValidCol, [&] {
      rmsnorm_apply<tileVec, tileVec, tileVec, tileSum>(
          tOut.data(), tX.data(), tGamma.data(), tInv.data());
    });
    TSTORE(gO(token, 0), tOut);
  }
}

} // namespace kernels
} // namespace pto

#endif // PTO_COMMON_BLOCK_VECTOR_KERNELS_HPP
