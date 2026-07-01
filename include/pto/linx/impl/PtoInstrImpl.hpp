#ifndef PTO_LINX_IMPL_PTO_INSTR_IMPL_HPP
#define PTO_LINX_IMPL_PTO_INSTR_IMPL_HPP

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include <common/pto_tileop.hpp>
#include <pto/common/constants.hpp>
#include <pto/cpu/tile_offsets.hpp>

namespace pto {
namespace linx {
namespace impl {

template <typename... Ts>
struct dependent_false {
  static constexpr bool value = false;
};

template <typename... Args>
inline void Unsupported(const char *op_name) {
  (void)op_name;
  static_assert(dependent_false<Args...>::value,
                "LinxISA v0.57: unsupported PTO op for __LINXISA__ backend");
}

template <typename IndexT>
inline std::size_t ConcatClampIndex(IndexT raw, std::size_t limit) {
  static_assert(std::is_integral_v<IndexT>,
                "TCONCATIDX: index tiles must use an integral element type");
  if constexpr (std::is_signed_v<IndexT>) {
    if (raw <= 0) {
      return 0;
    }
  }
  return std::min<std::size_t>(static_cast<std::size_t>(raw), limit);
}

template <typename DstTileData, typename SrcTileData>
inline std::size_t ConcatCopyRow(DstTileData &dst, SrcTileData &src,
                                 std::size_t row, std::size_t dstCol,
                                 std::size_t cols) {
  const std::size_t dstCols = static_cast<std::size_t>(dst.GetValidCol());
  const std::size_t srcCols = static_cast<std::size_t>(src.GetValidCol());
  if (dstCol >= dstCols) {
    return 0;
  }

  const std::size_t copyCols = std::min(cols, std::min(srcCols, dstCols - dstCol));
  for (std::size_t c = 0; c < copyCols; ++c) {
    dst.data()[GetTileElementOffset<DstTileData>(row, dstCol + c)] =
        static_cast<typename DstTileData::DType>(
            src.data()[GetTileElementOffset<SrcTileData>(row, c)]);
  }
  return copyCols;
}

} // namespace impl
} // namespace linx

template <typename Dst, typename Src>
inline void TLOAD_IMPL(Dst &dst, Src &src) {
  TLOAD(dst, src);
}

template <typename TileData, typename GlobalData,
          AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu>
inline void TSTORE_IMPL(GlobalData &dst, TileData &src) {
  (void)atomicType;
  (void)reluPreMode;
  TSTORE(dst, src);
}

template <typename TileData, typename GlobalData, typename FpTileData,
          AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu>
inline void TSTORE_IMPL(GlobalData &dst, TileData &src, FpTileData &) {
  (void)atomicType;
  (void)reluPreMode;
  TSTORE(dst, src);
}

template <typename TileData, typename GlobalData,
          AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu>
inline void TSTORE_IMPL(GlobalData &dst, TileData &src, uint64_t) {
  (void)atomicType;
  (void)reluPreMode;
  TSTORE(dst, src);
}

template <typename TileRes, typename TileLeft, typename TileRight>
inline void TMATMUL_IMPL(TileRes &dst, TileLeft &lhs, TileRight &rhs) {
  TMATMUL(dst, lhs, rhs);
}

template <typename TileRes, typename TileLeft, typename TileRight>
inline void TMATMUL_ACC_IMPL(TileRes &dst, TileRes &acc, TileLeft &lhs,
                             TileRight &rhs) {
  TMATMUL_ACC(dst, acc, lhs, rhs);
}

template <typename Dst, typename Src0, typename Src1>
inline void TADD_IMPL(Dst &dst, Src0 &src0, Src1 &src1) {
  TADD(dst, src0, src1);
}

template <typename Dst, typename Src0, typename Src1>
inline void TSUB_IMPL(Dst &dst, Src0 &src0, Src1 &src1) {
  TSUB(dst, src0, src1);
}

template <typename Dst, typename Src0, typename Src1>
inline void TMUL_IMPL(Dst &dst, Src0 &src0, Src1 &src1) {
  TMUL(dst, src0, src1);
}

template <typename Dst, typename Src0, typename Src1>
inline void TMAX_IMPL(Dst &dst, Src0 &src0, Src1 &src1) {
  TMAX(dst, src0, src1);
}

template <typename Dst, typename Src>
inline void TEXPANDS_IMPL(Dst &dst, typename Dst::DType scalar) {
  TEXPANDS(dst, scalar);
}

template <typename Dst, typename Src>
inline void TEXP_IMPL(Dst &dst, Src &src) {
  TEXP(dst, src);
}

template <typename Dst, typename Src>
inline void TRECIP_IMPL(Dst &dst, Src &src) {
  TRECIP(dst, src);
}

template <typename Dst, typename Src0, typename Src1>
inline void TCONCAT_IMPL(Dst &dst, Src0 &src0, Src1 &src1) {
  static_assert(std::is_same_v<typename Dst::DType, typename Src0::DType>,
                "TCONCAT: dst and src0 must use the same element type");
  static_assert(std::is_same_v<typename Dst::DType, typename Src1::DType>,
                "TCONCAT: dst and src1 must use the same element type");

  const std::size_t dstRows = static_cast<std::size_t>(dst.GetValidRow());
  const std::size_t dstCols = static_cast<std::size_t>(dst.GetValidCol());
  const std::size_t src0Rows = static_cast<std::size_t>(src0.GetValidRow());
  const std::size_t src0Cols = static_cast<std::size_t>(src0.GetValidCol());
  const std::size_t src1Rows = static_cast<std::size_t>(src1.GetValidRow());
  const std::size_t src1Cols = static_cast<std::size_t>(src1.GetValidCol());
  const std::size_t rows = std::min(dstRows, std::min(src0Rows, src1Rows));

  for (std::size_t r = 0; r < rows; ++r) {
    std::size_t dstCol = linx::impl::ConcatCopyRow(dst, src0, r, 0, src0Cols);
    if (dstCol < dstCols) {
      linx::impl::ConcatCopyRow(dst, src1, r, dstCol, src1Cols);
    }
  }
}

template <typename Dst, typename Src0, typename Src1, typename Src0Idx,
          typename Src1Idx>
inline void TCONCATIDX_IMPL(Dst &dst, Src0 &src0, Src1 &src1, Src0Idx &src0Idx,
                            Src1Idx &src1Idx) {
  static_assert(std::is_same_v<typename Dst::DType, typename Src0::DType>,
                "TCONCATIDX: dst and src0 must use the same element type");
  static_assert(std::is_same_v<typename Dst::DType, typename Src1::DType>,
                "TCONCATIDX: dst and src1 must use the same element type");
  static_assert(std::is_integral_v<typename Src0Idx::DType>,
                "TCONCATIDX: src0Idx must use an integral element type");
  static_assert(std::is_integral_v<typename Src1Idx::DType>,
                "TCONCATIDX: src1Idx must use an integral element type");

  if (src0Idx.GetValidCol() == 0 || src1Idx.GetValidCol() == 0) {
    return;
  }

  const std::size_t dstRows = static_cast<std::size_t>(dst.GetValidRow());
  const std::size_t dstCols = static_cast<std::size_t>(dst.GetValidCol());
  const std::size_t src0Rows = static_cast<std::size_t>(src0.GetValidRow());
  const std::size_t src0Cols = static_cast<std::size_t>(src0.GetValidCol());
  const std::size_t src1Rows = static_cast<std::size_t>(src1.GetValidRow());
  const std::size_t src1Cols = static_cast<std::size_t>(src1.GetValidCol());
  const std::size_t idx0Rows = static_cast<std::size_t>(src0Idx.GetValidRow());
  const std::size_t idx1Rows = static_cast<std::size_t>(src1Idx.GetValidRow());
  const std::size_t rows =
      std::min(std::min(dstRows, src0Rows), std::min(src1Rows, std::min(idx0Rows, idx1Rows)));

  for (std::size_t r = 0; r < rows; ++r) {
    const auto idx0Raw = src0Idx.data()[GetTileElementOffset<Src0Idx>(r, 0)];
    const auto idx1Raw = src1Idx.data()[GetTileElementOffset<Src1Idx>(r, 0)];
    const std::size_t idx0Cols = linx::impl::ConcatClampIndex(idx0Raw, src0Cols);
    const std::size_t idx1Cols = linx::impl::ConcatClampIndex(idx1Raw, src1Cols);

    const std::size_t dstCol = linx::impl::ConcatCopyRow(
        dst, src0, r, 0, std::min(idx0Cols, dstCols));
    if (dstCol < dstCols) {
      linx::impl::ConcatCopyRow(dst, src1, r, dstCol, std::min(idx1Cols, dstCols - dstCol));
    }
  }
}

template <typename Dst, typename Src>
inline void TCOLEXPAND_IMPL(Dst &dst, Src &src) {
  TCOLEXPAND(dst, src);
}

template <typename Dst, typename Src>
inline void TCVT_IMPL(Dst &dst, Src &src, RoundMode) {
  TCVT(dst, src);
}

template <typename Dst, typename Src, typename Tmp>
inline void TROWSUM_IMPL(Dst &dst, Src &src, Tmp &) {
  TROWSUM(dst, src);
}

template <typename Dst, typename Src, typename Tmp>
inline void TROWMAX_IMPL(Dst &dst, Src &src, Tmp &) {
  TROWMAX(dst, src);
}

template <typename Dst, typename Src>
inline void TMULS_IMPL(Dst &dst, Src &src, typename Src::DType scalar) {
  TMULS(dst, src, scalar);
}

template <typename Dst, typename Scalar, typename Src>
inline void TDIVS_IMPL(Dst &dst, Scalar, Src &src) {
  TRECIP(dst, src);
}

template <typename... Args>
inline void TAND_IMPL(Args &&...) {
  linx::impl::Unsupported<Args...>("TAND");
}

} // namespace pto

#endif // PTO_LINX_IMPL_PTO_INSTR_IMPL_HPP
