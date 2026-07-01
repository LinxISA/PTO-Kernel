/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCONCAT_HPP
#define TCONCAT_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {

template <typename IndexT>
PTO_INLINE std::size_t TCONCAT_CLAMP_INDEX(IndexT raw, std::size_t limit)
{
    static_assert(std::is_integral_v<IndexT>, "TCONCATIDX: index tiles must use an integral element type");
    if constexpr (std::is_signed_v<IndexT>) {
        if (raw <= 0) {
            return 0;
        }
    }
    return std::min<std::size_t>(static_cast<std::size_t>(raw), limit);
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL std::size_t TCONCAT_COPY_ROW(DstTileData &dst, SrcTileData &src, std::size_t row,
    std::size_t dstCol, std::size_t cols)
{
    const std::size_t dstCols = static_cast<std::size_t>(dst.GetValidCol());
    const std::size_t srcCols = static_cast<std::size_t>(src.GetValidCol());
    if (dstCol >= dstCols) {
        return 0;
    }
    const std::size_t copyCols = std::min(cols, std::min(srcCols, dstCols - dstCol));
    for (std::size_t c = 0; c < copyCols; ++c) {
        dst.data()[GetTileElementOffset<DstTileData>(row, dstCol + c)] =
            static_cast<typename DstTileData::DType>(src.data()[GetTileElementOffset<SrcTileData>(row, c)]);
    }
    return copyCols;
}

template <typename DstTileData, typename Src0TileData, typename Src1TileData>
PTO_INTERNAL void TCONCAT_IMPL(DstTileData &dst, Src0TileData &src0, Src1TileData &src1)
{
    static_assert(std::is_same_v<typename DstTileData::DType, typename Src0TileData::DType>,
        "TCONCAT: dst and src0 must use the same element type");
    static_assert(std::is_same_v<typename DstTileData::DType, typename Src1TileData::DType>,
        "TCONCAT: dst and src1 must use the same element type");

    const std::size_t dstRows = static_cast<std::size_t>(dst.GetValidRow());
    const std::size_t dstCols = static_cast<std::size_t>(dst.GetValidCol());
    const std::size_t src0Rows = static_cast<std::size_t>(src0.GetValidRow());
    const std::size_t src0Cols = static_cast<std::size_t>(src0.GetValidCol());
    const std::size_t src1Rows = static_cast<std::size_t>(src1.GetValidRow());
    const std::size_t src1Cols = static_cast<std::size_t>(src1.GetValidCol());

    assert(src0Rows == dstRows);
    assert(src1Rows == dstRows);
    assert(src0Cols + src1Cols <= dstCols);

    const std::size_t rows = std::min(dstRows, std::min(src0Rows, src1Rows));
    for (std::size_t r = 0; r < rows; ++r) {
        std::size_t dstCol = TCONCAT_COPY_ROW(dst, src0, r, 0, src0Cols);
        if (dstCol < dstCols) {
            TCONCAT_COPY_ROW(dst, src1, r, dstCol, src1Cols);
        }
    }
}

template <typename DstTileData, typename Src0TileData, typename Src1TileData,
          typename Src0IdxTileData, typename Src1IdxTileData>
PTO_INTERNAL void TCONCATIDX_IMPL(DstTileData &dst, Src0TileData &src0, Src1TileData &src1,
    Src0IdxTileData &src0Idx, Src1IdxTileData &src1Idx)
{
    static_assert(std::is_same_v<typename DstTileData::DType, typename Src0TileData::DType>,
        "TCONCATIDX: dst and src0 must use the same element type");
    static_assert(std::is_same_v<typename DstTileData::DType, typename Src1TileData::DType>,
        "TCONCATIDX: dst and src1 must use the same element type");
    static_assert(std::is_integral_v<typename Src0IdxTileData::DType>,
        "TCONCATIDX: src0Idx must use an integral element type");
    static_assert(std::is_integral_v<typename Src1IdxTileData::DType>,
        "TCONCATIDX: src1Idx must use an integral element type");

    const std::size_t dstRows = static_cast<std::size_t>(dst.GetValidRow());
    const std::size_t dstCols = static_cast<std::size_t>(dst.GetValidCol());
    const std::size_t src0Rows = static_cast<std::size_t>(src0.GetValidRow());
    const std::size_t src0Cols = static_cast<std::size_t>(src0.GetValidCol());
    const std::size_t src1Rows = static_cast<std::size_t>(src1.GetValidRow());
    const std::size_t src1Cols = static_cast<std::size_t>(src1.GetValidCol());
    const std::size_t idx0Rows = static_cast<std::size_t>(src0Idx.GetValidRow());
    const std::size_t idx1Rows = static_cast<std::size_t>(src1Idx.GetValidRow());

    assert(src0Rows == dstRows);
    assert(src1Rows == dstRows);
    assert(idx0Rows == dstRows);
    assert(idx1Rows == dstRows);
    assert(src0Idx.GetValidCol() > 0);
    assert(src1Idx.GetValidCol() > 0);

    if (src0Idx.GetValidCol() == 0 || src1Idx.GetValidCol() == 0) {
        return;
    }

    const std::size_t rows = std::min(std::min(dstRows, src0Rows), std::min(src1Rows, std::min(idx0Rows, idx1Rows)));
    for (std::size_t r = 0; r < rows; ++r) {
        const auto idx0Raw = src0Idx.data()[GetTileElementOffset<Src0IdxTileData>(r, 0)];
        const auto idx1Raw = src1Idx.data()[GetTileElementOffset<Src1IdxTileData>(r, 0)];
        const std::size_t idx0Cols = TCONCAT_CLAMP_INDEX(idx0Raw, src0Cols);
        const std::size_t idx1Cols = TCONCAT_CLAMP_INDEX(idx1Raw, src1Cols);

        std::size_t dstCol = TCONCAT_COPY_ROW(dst, src0, r, 0, std::min(idx0Cols, dstCols));
        if (dstCol < dstCols) {
            TCONCAT_COPY_ROW(dst, src1, r, dstCol, std::min(idx1Cols, dstCols - dstCol));
        }
    }
}

} // namespace pto

#endif // TCONCAT_HPP
