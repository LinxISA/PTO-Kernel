#ifndef PTO_COMMON_PTO_TILEOP_HPP
#define PTO_COMMON_PTO_TILEOP_HPP

#include <stddef.h>
#include <stdint.h>

#include <pto/linx/impl/backend.hpp>

namespace pto {

enum class Location : uint8_t {
  Vec,
  Left,
  Right,
  Acc,
  Scaling,
};

enum class BLayout : uint8_t {
  RowMajor = 0,
  ColMajor = 1,
};

enum class SLayout : uint8_t {
  NoneBox = 0,
  RowMajor = 1,
  ColMajor = 2,
};

enum class CmpMode : uint8_t {
  EQ = 0,
  NE = 1,
  LT = 2,
  LE = 3,
  GT = 4,
  GE = 5,
};

namespace TileConfig {
static constexpr int alignedSize = 32;
static constexpr int fixedRowSize = 16;
static constexpr int fixedColSize = 16;
static constexpr int fixedMxRowSize = 16;
static constexpr int fixedMxColSize = 2;
static constexpr int fractalABSize = 512;
static constexpr int fractalCSize = 1024;
static constexpr int cElemSize = 4;
} // namespace TileConfig

template <int Rows_, int Cols_>
struct RowMajor {
  static constexpr int Rows = Rows_;
  static constexpr int Cols = Cols_;
  static constexpr bool IsRowMajor = true;
};

template <int Rows_, int Cols_>
struct ColMajor {
  static constexpr int Rows = Rows_;
  static constexpr int Cols = Cols_;
  static constexpr bool IsRowMajor = false;
};

template <typename Element_, typename Layout_>
struct global_tensor {
  using DType = Element_;
  using Layout = Layout_;

  explicit global_tensor(Element_ *base) : base_(base) {}
  global_tensor(Element_ *base, int, int) : base_(base) {}

  Element_ *data() const { return base_; }
  Element_ *ptr() const { return base_; }

private:
  Element_ *base_;
};

template <int Dim0_, int Dim1_, int Dim2_, int Dim3_, int Dim4_>
struct Shape {
  static constexpr int Dim0 = Dim0_;
  static constexpr int Dim1 = Dim1_;
  static constexpr int Dim2 = Dim2_;
  static constexpr int Dim3 = Dim3_;
  static constexpr int Dim4 = Dim4_;
  static constexpr int Rows = Dim3_;
  static constexpr int Cols = Dim4_;
};

template <int Dim0_, int Dim1_, int Dim2_, int Dim3_, int Dim4_>
struct Stride {
  static constexpr int Dim0 = Dim0_;
  static constexpr int Dim1 = Dim1_;
  static constexpr int Dim2 = Dim2_;
  static constexpr int Dim3 = Dim3_;
  static constexpr int Dim4 = Dim4_;
  static constexpr bool IsRowMajor = Dim4_ == 1;
};

enum class GlobalTensorDim : uint8_t {
  DIM_0 = 0,
  DIM_1 = 1,
  DIM_2 = 2,
  DIM_3 = 3,
  DIM_4 = 4,
};

template <typename Element_, typename Shape_, typename Stride_>
struct GlobalTensor {
  using DType = Element_;
  using ShapeT = Shape_;
  using StrideT = Stride_;
  struct Layout {
    static constexpr int Rows = Shape_::Rows;
    static constexpr int Cols = Shape_::Cols;
    static constexpr bool IsRowMajor = Stride_::IsRowMajor;
  };

  explicit GlobalTensor(Element_ *base) : base_(base) {}

  Element_ *data() const { return base_; }
  Element_ *ptr() const { return base_; }

  static constexpr int GetShape(GlobalTensorDim dim) {
    switch (dim) {
    case GlobalTensorDim::DIM_0:
      return Shape_::Dim0;
    case GlobalTensorDim::DIM_1:
      return Shape_::Dim1;
    case GlobalTensorDim::DIM_2:
      return Shape_::Dim2;
    case GlobalTensorDim::DIM_3:
      return Shape_::Dim3;
    case GlobalTensorDim::DIM_4:
      return Shape_::Dim4;
    }
    return 0;
  }

  static constexpr int GetStride(GlobalTensorDim dim) {
    switch (dim) {
    case GlobalTensorDim::DIM_0:
      return Stride_::Dim0;
    case GlobalTensorDim::DIM_1:
      return Stride_::Dim1;
    case GlobalTensorDim::DIM_2:
      return Stride_::Dim2;
    case GlobalTensorDim::DIM_3:
      return Stride_::Dim3;
    case GlobalTensorDim::DIM_4:
      return Stride_::Dim4;
    }
    return 0;
  }

private:
  Element_ *base_;
};

namespace detail {

using ptrdiff_builtin_t = __PTRDIFF_TYPE__;

template <typename... Ts>
using void_t = void;

template <bool Value>
struct bool_constant {
  static constexpr bool value = Value;
};

using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template <typename T>
T &&declval() noexcept;

// TMA format selectors used by B.ARG in the LinxISA v0.57 profile.
constexpr long long kLayoutNorm = 0ll;     // NORM.normal
constexpr long long kLayoutND2NZ = 2ll;    // ND2NZ.normal
constexpr long long kLayoutND2ZN = 3ll;    // ND2ZN.normal
constexpr long long kLayoutDN2ZN = 8ll;    // DN2ZN.normal
constexpr long long kLayoutDN2NZ = 9ll;    // DN2NZ.normal

template <typename TileT>
constexpr unsigned tileBytes() {
  constexpr int rows = TileT::Rows;
  constexpr int cols = TileT::Cols;
  constexpr unsigned bytes =
      static_cast<unsigned>(rows * cols * sizeof(typename TileT::DType));
  static_assert(bytes > 0u, "PTO Linx v0.57: tile bytes must be positive");
  return bytes;
}

template <typename TileT>
constexpr unsigned tileSizeCode() {
  static_assert(tileBytes<TileT>() <= linx::detail::kMaxTileBytes,
                "PTO Linx v0.57: tile size exceeds 4KB");
  // Keep a single 4KB size profile in PR5 user-facing wrappers to avoid
  // cross-op metadata skew while strict Tile SSA balancing is enabled.
  return 8u;
}

template <typename TileT>
constexpr unsigned tileDTypeCode() {
  return linx::detail::DTypeCode<typename TileT::DType>::value;
}

template <typename TileT>
constexpr long long tileLayoutCode() {
  return TileT::LayoutTag == BLayout::RowMajor ? 0ll : 1ll;
}

template <typename GTensor>
constexpr long long gmStrideBytes() {
  constexpr long long elemBytes =
      static_cast<long long>(sizeof(typename GTensor::DType));
  if constexpr (GTensor::Layout::IsRowMajor)
    return static_cast<long long>(GTensor::Layout::Cols) * elemBytes;
  return static_cast<long long>(GTensor::Layout::Rows) * elemBytes;
}

template <typename GTensor, typename TileT>
constexpr long long tensorTileLayoutCode() {
  if constexpr (TileT::Loc == Location::Left || TileT::Loc == Location::Acc) {
    return GTensor::Layout::IsRowMajor ? kLayoutND2ZN : kLayoutDN2ZN;
  }
  if constexpr (TileT::Loc == Location::Right) {
    return GTensor::Layout::IsRowMajor ? kLayoutND2NZ : kLayoutDN2NZ;
  }
  return kLayoutNorm;
}

template <typename TileT>
constexpr long long tileLB0() {
  return TileT::RowValid > 0 ? static_cast<long long>(TileT::RowValid)
                             : static_cast<long long>(TileT::Rows);
}

template <typename TileT>
constexpr long long tileLB1() {
  return TileT::ColValid > 0 ? static_cast<long long>(TileT::ColValid)
                             : static_cast<long long>(TileT::Cols);
}

template <typename TileT>
constexpr long long tileGmLB0() {
  return TileT::LayoutTag == BLayout::RowMajor ? tileLB1<TileT>()
                                               : tileLB0<TileT>();
}

template <typename TileT>
constexpr long long tileGmLB1() {
  return TileT::LayoutTag == BLayout::RowMajor ? tileLB0<TileT>()
                                               : tileLB1<TileT>();
}

template <typename GTensor, typename TileT>
inline ptrdiff_builtin_t tileOffset(int tileRow, int tileCol) {
  const int row = tileRow * TileT::Rows;
  const int col = tileCol * TileT::Cols;
  if constexpr (GTensor::Layout::IsRowMajor) {
    return static_cast<ptrdiff_builtin_t>(row) * GTensor::Layout::Cols + col;
  }
  return static_cast<ptrdiff_builtin_t>(col) * GTensor::Layout::Rows + row;
}

template <typename AddressLike>
inline auto addressPtr(const AddressLike &addr) -> decltype(addr.ptr()) {
  return addr.ptr();
}

template <typename T>
inline T *addressPtr(T *addr) {
  return addr;
}

template <typename T>
inline const T *addressPtr(const T *addr) {
  return addr;
}

template <typename AddressLike, typename TileT, typename = void>
struct AddressDesc {
  static constexpr long long Layout = kLayoutNorm;
  static constexpr long long LB0 = tileGmLB0<TileT>();
  static constexpr long long LB1 = tileGmLB1<TileT>();
  static constexpr long long StrideBytes = 0ll;
};

template <typename AddressLike, typename TileT>
struct AddressDesc<AddressLike, TileT,
                   void_t<decltype(AddressLike::kLayoutCode),
                          decltype(AddressLike::kLB0),
                          decltype(AddressLike::kLB1),
                          decltype(AddressLike::kStrideBytes)>> {
  static constexpr long long Layout = AddressLike::kLayoutCode;
  static constexpr long long LB0 = AddressLike::kLB0;
  static constexpr long long LB1 = AddressLike::kLB1;
  static constexpr long long StrideBytes = AddressLike::kStrideBytes;
};

template <typename AddressLike, typename TileT>
constexpr long long addressLayoutCode() {
  return AddressDesc<AddressLike, TileT>::Layout;
}

template <typename AddressLike, typename TileT>
constexpr long long addressLB0() {
  return AddressDesc<AddressLike, TileT>::LB0;
}

template <typename AddressLike, typename TileT>
constexpr long long addressLB1() {
  return AddressDesc<AddressLike, TileT>::LB1;
}

template <typename AddressLike, typename TileT>
constexpr long long addressStrideBytes() {
  return AddressDesc<AddressLike, TileT>::StrideBytes;
}

} // namespace detail

template <Location Loc_, typename Element_, int Rows_, int Cols_,
          BLayout Layout_ = BLayout::RowMajor, int RowValid_ = Rows_,
          int ColValid_ = Cols_, SLayout SFractal_ = SLayout::NoneBox>
struct Tile {
  using DType = Element_;
  using RawTile = linx::detail::RawTile;
  using TileDType = Tile *;
  using ConstTileDType = const Tile *;

  static constexpr int getInnerRow() {
    if constexpr (SFractal_ == SLayout::NoneBox) {
      return 1;
    } else if constexpr (SFractal_ == SLayout::RowMajor) {
      return TileConfig::fixedRowSize;
    } else {
      return TileConfig::alignedSize / static_cast<int>(sizeof(DType));
    }
  }

  static constexpr int getInnerCol() {
    if constexpr (SFractal_ == SLayout::NoneBox) {
      return 1;
    } else if constexpr (SFractal_ == SLayout::RowMajor) {
      return TileConfig::alignedSize / static_cast<int>(sizeof(DType));
    } else {
      return TileConfig::fixedColSize;
    }
  }

  static constexpr Location Loc = Loc_;
  static constexpr int Rows = Rows_;
  static constexpr int Cols = Cols_;
  static constexpr int RowValid = RowValid_;
  static constexpr int ColValid = ColValid_;
  static constexpr int ValidRow = RowValid_;
  static constexpr int ValidCol = ColValid_;
  static constexpr BLayout LayoutTag = Layout_;
  static constexpr BLayout BFractal = Layout_;
  static constexpr SLayout SFractal = SFractal_;
  static constexpr int Numel = Rows_ * Cols_;
  static constexpr bool isRowMajor = Layout_ == BLayout::RowMajor;
  static constexpr bool isBoxedLayout = SFractal_ != SLayout::NoneBox;
  static constexpr bool isInnerRowMajor = SFractal_ == SLayout::RowMajor;
  static constexpr bool isInnerColMajor = SFractal_ == SLayout::ColMajor;
  static constexpr int InnerRows = getInnerRow();
  static constexpr int InnerCols = getInnerCol();
  static constexpr int InnerNumel = InnerRows * InnerCols;
  static constexpr int RowStride =
      LayoutTag == BLayout::RowMajor ? Cols_ : 1;
  static constexpr int ColStride =
      LayoutTag == BLayout::RowMajor ? 1 : Rows_;

  Tile() = default;

  template <typename Scalar>
  explicit Tile(Scalar scalar) {
    raw_ = linx::detail::teplSplat<0x01du, detail::tileSizeCode<Tile>(),
                                   detail::tileDTypeCode<Tile>(), 2u>(scalar);
  }

  RawTile &raw() { return raw_; }
  const RawTile &raw() const { return raw_; }
  TileDType data() { return this; }
  ConstTileDType data() const { return this; }
  static constexpr int GetValidRow() { return RowValid_; }
  static constexpr int GetValidCol() { return ColValid_; }

private:
  RawTile raw_{};
};

template <typename Element_, int Rows_, int Cols_, int RowValid_ = Rows_,
          int ColValid_ = Cols_>
using TileLeft =
    Tile<Location::Left, Element_, Rows_, Cols_, BLayout::ColMajor, RowValid_,
         ColValid_>;

template <typename Element_, int Rows_, int Cols_, int RowValid_ = Rows_,
          int ColValid_ = Cols_>
using TileRight =
    Tile<Location::Right, Element_, Rows_, Cols_, BLayout::RowMajor, RowValid_,
         ColValid_>;

template <typename Element_, int Rows_, int Cols_, int RowValid_ = Rows_,
          int ColValid_ = Cols_>
using TileAcc =
    Tile<Location::Acc, Element_, Rows_, Cols_, BLayout::ColMajor, RowValid_,
         ColValid_>;

template <typename GTensor, typename TileT>
class global_iterator {
public:
  using Element = typename GTensor::DType;

  explicit global_iterator(Element *base) : base_(base) {}

  struct tile_address {
    using TensorType = GTensor;
    using TileType = TileT;
    static constexpr long long kLayoutCode =
        detail::tensorTileLayoutCode<GTensor, TileT>();
    // TMA contract: LB0/LB1 are GM-side inner/outer counts.
    // ND(row-major): inner=cols, outer=rows; DN(column-major): inner=rows, outer=cols.
    static constexpr long long kLB0 =
        GTensor::Layout::IsRowMajor ? detail::tileLB1<TileT>()
                                    : detail::tileLB0<TileT>();
    static constexpr long long kLB1 =
        GTensor::Layout::IsRowMajor ? detail::tileLB0<TileT>()
                                    : detail::tileLB1<TileT>();
    static constexpr long long kStrideBytes = detail::gmStrideBytes<GTensor>();

    Element *base;
    int tileRow;
    int tileCol;

    Element *ptr() const {
      return base + detail::tileOffset<GTensor, TileT>(tileRow, tileCol);
    }
  };

  tile_address operator()(int tileRow, int tileCol) const {
    return tile_address{base_, tileRow, tileCol};
  }

private:
  Element *base_;
};

template <typename T>
struct tile_storage_type {
  using type = T;
};

template <typename T>
struct tile_storage_type<T *> {
  using type = T;
};

template <typename T>
struct tile_storage_type<const T *> {
  using type = T;
};

template <typename T>
struct type_traits {
  static constexpr unsigned TypeCode = linx::detail::DTypeCode<T>::value;
  static constexpr unsigned bits =
      linx::detail::dtypeElemBits(linx::detail::DTypeCode<T>::value);
};

template <typename TileT>
struct tile_type_traits {
  using Tile = typename tile_storage_type<TileT>::type;
  static constexpr unsigned TilesizeCode = detail::tileSizeCode<Tile>();
};

namespace detail {

template <typename T, typename = void>
struct is_tile_data_impl : false_type {};

template <typename T>
struct is_tile_data_impl<
    T, void_t<typename T::DType, typename T::TileDType, typename T::RawTile,
              decltype(T::Rows), decltype(T::Cols), decltype(T::ValidRow),
              decltype(T::ValidCol), decltype(declval<T &>().data()),
              decltype(declval<T &>().raw())>> : true_type {};

template <typename T, typename = void>
struct has_global_tensor_data : false_type {};

template <typename T>
struct has_global_tensor_data<
    T, void_t<typename T::DType, typename T::Layout,
              decltype(declval<T &>().data())>> : true_type {};

template <typename T, typename = void>
struct has_global_tile_address : false_type {};

template <typename T>
struct has_global_tile_address<
    T, void_t<typename T::TensorType, typename T::TileType,
              decltype(T::kLayoutCode), decltype(T::kLB0), decltype(T::kLB1),
              decltype(T::kStrideBytes), decltype(declval<T &>().ptr())>>
    : true_type {};

template <typename T>
struct is_global_data_impl
    : bool_constant<has_global_tensor_data<T>::value ||
                    has_global_tile_address<T>::value> {};

} // namespace detail

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
template <typename T>
concept is_tile_data_v = detail::is_tile_data_impl<T>::value;

template <typename T>
concept is_global_data_v = detail::is_global_data_impl<T>::value;
#else
template <typename T>
constexpr bool is_tile_data_v = detail::is_tile_data_impl<T>::value;

template <typename T>
constexpr bool is_global_data_v = detail::is_global_data_impl<T>::value;
#endif

namespace tepl {
constexpr unsigned TABS = 0x000u;
constexpr unsigned TADD = 0x001u;
constexpr unsigned TADDC = 0x002u;
constexpr unsigned TADDS = 0x003u;
constexpr unsigned TADDSC = 0x004u;
constexpr unsigned TAND = 0x005u;
constexpr unsigned TANDS = 0x006u;
constexpr unsigned TAXPY = 0x007u;
constexpr unsigned TCI = 0x008u;
constexpr unsigned TCMP = 0x009u;
constexpr unsigned TCMPS = 0x00au;
constexpr unsigned TCOLARGMAX = 0x00bu;
constexpr unsigned TCOLARGMIN = 0x00cu;
constexpr unsigned TCOLEXPAND = 0x00du;
constexpr unsigned TCOLEXPANDADD = 0x00eu;
constexpr unsigned TCOLEXPANDDIV = 0x00fu;
constexpr unsigned TCOLEXPANDEXPDIF = 0x010u;
constexpr unsigned TCOLEXPANDMAX = 0x011u;
constexpr unsigned TCOLEXPANDMIN = 0x012u;
constexpr unsigned TCOLEXPANDMUL = 0x013u;
constexpr unsigned TCOLEXPANDSUB = 0x014u;
constexpr unsigned TCOLMAX = 0x015u;
constexpr unsigned TCOLMIN = 0x016u;
constexpr unsigned TCOLPROD = 0x017u;
constexpr unsigned TCOLSUM = 0x018u;
constexpr unsigned TCVT = 0x019u;
constexpr unsigned TDIV = 0x01au;
constexpr unsigned TDIVS = 0x01bu;
constexpr unsigned TEXP = 0x01cu;
constexpr unsigned TEXPANDS = 0x01du;
constexpr unsigned TFMOD = 0x01eu;
constexpr unsigned TFMODS = 0x01fu;
constexpr unsigned TGATHER = 0x020u;
constexpr unsigned TGATHERB = 0x021u;
constexpr unsigned THISTOGRAM = 0x022u;
constexpr unsigned TLOG = 0x023u;
constexpr unsigned TLRELU = 0x024u;
constexpr unsigned TMAX = 0x025u;
constexpr unsigned TMAXS = 0x026u;
constexpr unsigned TMIN = 0x027u;
constexpr unsigned TMINS = 0x028u;
constexpr unsigned TMRGSORT = 0x029u;
constexpr unsigned TMUL = 0x02au;
constexpr unsigned TMULS = 0x02bu;
constexpr unsigned TNEG = 0x02cu;
constexpr unsigned TNOT = 0x02du;
constexpr unsigned TOR = 0x02eu;
constexpr unsigned TORS = 0x02fu;
constexpr unsigned TPARTADD = 0x030u;
constexpr unsigned TPARTARGMAX = 0x031u;
constexpr unsigned TPARTARGMIN = 0x032u;
constexpr unsigned TPARTMAX = 0x033u;
constexpr unsigned TPARTMIN = 0x034u;
constexpr unsigned TPARTMUL = 0x035u;
constexpr unsigned TPOW = 0x036u;
constexpr unsigned TPRELU = 0x037u;
constexpr unsigned TRANDOM = 0x038u;
constexpr unsigned TRECIP = 0x039u;
constexpr unsigned TRELU = 0x03au;
constexpr unsigned TREM = 0x03bu;
constexpr unsigned TREMS = 0x03cu;
constexpr unsigned TROWARGMAX = 0x03du;
constexpr unsigned TROWARGMIN = 0x03eu;
constexpr unsigned TROWEXPAND = 0x03fu;
constexpr unsigned TROWEXPANDADD = 0x040u;
constexpr unsigned TROWEXPANDDIV = 0x041u;
constexpr unsigned TROWEXPANDEXPDIF = 0x042u;
constexpr unsigned TROWEXPANDMAX = 0x043u;
constexpr unsigned TROWEXPANDMIN = 0x044u;
constexpr unsigned TROWEXPANDMUL = 0x045u;
constexpr unsigned TROWEXPANDSUB = 0x046u;
constexpr unsigned TROWMAX = 0x047u;
constexpr unsigned TROWMIN = 0x048u;
constexpr unsigned TROWPROD = 0x049u;
constexpr unsigned TROWSUM = 0x04au;
constexpr unsigned TRSQRT = 0x04bu;
constexpr unsigned TSCATTER = 0x04cu;
constexpr unsigned TSEL = 0x04du;
constexpr unsigned TSELS = 0x04eu;
constexpr unsigned TSHL = 0x04fu;
constexpr unsigned TSHLS = 0x050u;
constexpr unsigned TSHR = 0x051u;
constexpr unsigned TSHRS = 0x052u;
constexpr unsigned TSORT32 = 0x053u;
constexpr unsigned TSQRT = 0x054u;
constexpr unsigned TSUB = 0x055u;
constexpr unsigned TSUBC = 0x056u;
constexpr unsigned TSUBS = 0x057u;
constexpr unsigned TSUBSC = 0x058u;
constexpr unsigned TTRI = 0x059u;
constexpr unsigned TXOR = 0x05au;
constexpr unsigned TXORS = 0x05bu;
} // namespace tepl

// Core tile ops used by PR5 FlashAttention bring-up.
template <typename DstTile, typename SrcAddress>
inline void TLOAD(DstTile &dst, const SrcAddress &src) {
  dst.raw() = linx::detail::tileTLoad<detail::tileSizeCode<DstTile>(),
                                      detail::tileDTypeCode<DstTile>(),
                                      detail::addressLayoutCode<SrcAddress, DstTile>(),
                                      detail::addressLB0<SrcAddress, DstTile>(),
                                      detail::addressLB1<SrcAddress, DstTile>(),
                                      detail::addressStrideBytes<SrcAddress, DstTile>()>(
      reinterpret_cast<const void *>(detail::addressPtr(src)));
}

template <typename DstAddress, typename SrcTile>
inline void TSTORE(const DstAddress &dst, SrcTile &src) {
  linx::detail::tileTStore<detail::tileSizeCode<SrcTile>(),
                           detail::tileDTypeCode<SrcTile>(),
                           detail::addressLayoutCode<DstAddress, SrcTile>(),
                           detail::addressLB0<DstAddress, SrcTile>(),
                           detail::addressLB1<DstAddress, SrcTile>(),
                           detail::addressStrideBytes<DstAddress, SrcTile>()>(
      reinterpret_cast<void *>(detail::addressPtr(dst)), src.raw());
}

template <typename DstAddress, typename SrcTile>
inline void TSTORE_FP(const DstAddress &dst, SrcTile &src) {
  TSTORE(dst, src);
}

template <typename DstTile, typename SrcAddress>
inline void TCOPYIN(DstTile &dst, const SrcAddress &src) {
  TLOAD(dst, src);
}

template <typename DstTile, typename Element_, typename Layout_>
inline void TCOPYIN(DstTile &dst,
                    const global_tensor<Element_, Layout_> &src) {
  dst.raw() = linx::detail::tileTLoad<detail::tileSizeCode<DstTile>(),
                                      detail::tileDTypeCode<DstTile>(),
                                      detail::kLayoutNorm,
                                      detail::tileGmLB0<DstTile>(),
                                      detail::tileGmLB1<DstTile>(),
                                      detail::gmStrideBytes<global_tensor<Element_, Layout_>>()>(
      reinterpret_cast<const void *>(src.data()));
}

template <typename DstTile, typename Element_, typename Layout_,
          typename IndexTile>
inline void MGATHER(DstTile &dst, const global_tensor<Element_, Layout_> &src,
                    const IndexTile &index) {
  dst.raw() = linx::detail::tileMGather<detail::tileSizeCode<DstTile>(),
                                        detail::tileDTypeCode<DstTile>(),
                                        detail::kLayoutNorm,
                                        detail::tileGmLB0<DstTile>(),
                                        detail::tileGmLB1<DstTile>(),
                                        detail::gmStrideBytes<
                                            global_tensor<Element_, Layout_>>()>(
      reinterpret_cast<const void *>(src.data()), index.raw());
}

template <typename DstTile, typename Element_, typename Shape_,
          typename Stride_, typename IndexTile>
inline void MGATHER(DstTile &dst,
                    const GlobalTensor<Element_, Shape_, Stride_> &src,
                    const IndexTile &index) {
  dst.raw() = linx::detail::tileMGather<detail::tileSizeCode<DstTile>(),
                                        detail::tileDTypeCode<DstTile>(),
                                        detail::kLayoutNorm,
                                        detail::tileGmLB0<DstTile>(),
                                        detail::tileGmLB1<DstTile>(),
                                        detail::gmStrideBytes<
                                            GlobalTensor<Element_, Shape_, Stride_>>()> (
      reinterpret_cast<const void *>(src.data()), index.raw());
}

template <typename DstAddress, typename SrcTile>
inline void TCOPYOUT(const DstAddress &dst, SrcTile &src) {
  TSTORE(dst, src);
}

template <typename Element_, typename Layout_, typename SrcTile>
inline void TCOPYOUT(const global_tensor<Element_, Layout_> &dst,
                     SrcTile &src) {
  linx::detail::tileTStore<detail::tileSizeCode<SrcTile>(),
                           detail::tileDTypeCode<SrcTile>(),
                           detail::kLayoutNorm, detail::tileGmLB0<SrcTile>(),
                           detail::tileGmLB1<SrcTile>(),
                           detail::gmStrideBytes<global_tensor<Element_, Layout_>>()>(
      reinterpret_cast<void *>(dst.data()), src.raw());
}

template <typename Element_, typename Layout_, typename SrcTile,
          typename IndexTile>
inline void MSCATTER(const global_tensor<Element_, Layout_> &dst, SrcTile &src,
                     const IndexTile &index) {
  linx::detail::tileMScatter<detail::tileSizeCode<SrcTile>(),
                             detail::tileDTypeCode<SrcTile>(),
                             detail::kLayoutNorm, detail::tileGmLB0<SrcTile>(),
                             detail::tileGmLB1<SrcTile>(),
                             detail::gmStrideBytes<
                                 global_tensor<Element_, Layout_>>()>(
      reinterpret_cast<void *>(dst.data()), src.raw(), index.raw());
}

template <typename DstTile, typename SrcTile>
inline void TMOV(DstTile &dst, const SrcTile &src, unsigned mode = 0u) {
  if (mode == 1u) {
    dst.raw() = linx::detail::tileTMov<detail::tileSizeCode<DstTile>(),
                                       detail::tileDTypeCode<DstTile>(),
                                       detail::tileLayoutCode<DstTile>(), 1u, 1u>(
        src.raw());
  } else {
    dst.raw() = linx::detail::tileTMov<detail::tileSizeCode<DstTile>(),
                                       detail::tileDTypeCode<DstTile>(),
                                       detail::tileLayoutCode<DstTile>(), 1u, 0u>(
        src.raw());
  }
}

template <typename DstTile, typename SrcTile>
inline void TINSERT(DstTile &dst, const SrcTile &src, uint16_t indexRow = 0,
                    uint16_t indexCol = 0) {
  const long long meta =
      (static_cast<long long>(indexRow) << 32) | static_cast<long long>(indexCol);
  dst.raw() = linx::detail::tileTInsert<detail::tileSizeCode<DstTile>(),
                                        detail::tileDTypeCode<DstTile>(),
                                        static_cast<unsigned>(DstTile::Rows),
                                        static_cast<unsigned>(DstTile::Cols),
                                        static_cast<unsigned>(SrcTile::Rows),
                                        static_cast<unsigned>(SrcTile::Cols)>(
      dst.raw(), src.raw(), meta);
}

template <typename DstTile, typename SrcTile, typename TmpTile>
inline void TTRANS(DstTile &dst, const SrcTile &src, TmpTile &tmp) {
  dst.raw() = linx::detail::tileTTrans<detail::tileSizeCode<DstTile>(),
                                       detail::tileDTypeCode<DstTile>(),
                                       static_cast<unsigned>(DstTile::Rows),
                                       static_cast<unsigned>(DstTile::Cols),
                                       static_cast<unsigned>(SrcTile::Rows),
                                       static_cast<unsigned>(SrcTile::Cols)>(
      src.raw(), tmp.raw());
}

template <typename DstTile, typename SrcTile>
inline void TTRANS(DstTile &dst, const SrcTile &src) {
  SrcTile tmp;
  TTRANS(dst, src, tmp);
}

template <typename TileRes, typename TileLeft_, typename TileRight_>
inline void TMATMUL(TileRes &dst, const TileLeft_ &lhs, const TileRight_ &rhs) {
  // LinxISA v0.57 compiler policy:
  // tile_bytes = ceil(m*n*k*elem_bits/8) must fit <=4KB
  // (m=Rows, n=Cols, k=lhs.Cols).
  constexpr unsigned M = static_cast<unsigned>(TileRes::Rows);
  constexpr unsigned N = static_cast<unsigned>(TileRes::Cols);
  constexpr unsigned K = static_cast<unsigned>(TileLeft_::Cols);
  dst.raw() = linx::detail::cubeMamulb<M, N, K>(lhs.raw(), rhs.raw());
}

template <typename TileRes, typename TileLeft_, typename TileRight_>
inline void MATMUL(TileRes &dst, const TileLeft_ &lhs, const TileRight_ &rhs) {
  TMATMUL(dst, lhs, rhs);
}

template <typename TileRes, typename TileLeft_, typename TileRight_>
inline void TMATMUL_ACC(TileRes &dst, TileRes &acc, const TileLeft_ &lhs,
                        const TileRight_ &rhs) {
  constexpr unsigned M = static_cast<unsigned>(TileRes::Rows);
  constexpr unsigned N = static_cast<unsigned>(TileRes::Cols);
  constexpr unsigned K = static_cast<unsigned>(TileLeft_::Cols);
  dst.raw() = linx::detail::cubeMamulbAcc<M, N, K>(acc.raw(), lhs.raw(), rhs.raw());
}

template <typename TileRes, typename TileLeft_, typename TileRight_>
inline void MATMACC(TileRes &dst, const TileLeft_ &lhs, const TileRight_ &rhs) {
  // Keep strict CUBE accumulator-chain legality: materialize the product with
  // TMATMUL, then accumulate explicitly with TEPL add.
  TileRes product;
  TMATMUL(product, lhs, rhs);
  TADD(dst, dst, product);
}

template <typename TileRes, typename TileLeft_, typename TileLeftScale_,
          typename TileRight_, typename TileRightScale_>
inline void MATMULMX(TileRes &dst, const TileLeft_ &lhs,
                     const TileLeftScale_ &, const TileRight_ &rhs,
                     const TileRightScale_ &) {
  TMATMUL(dst, lhs, rhs);
}

template <typename TileRes, typename TileLeft_, typename TileLeftScale_,
          typename TileRight_, typename TileRightScale_>
inline void MATMACCMX(TileRes &dst, const TileLeft_ &lhs,
                      const TileLeftScale_ &, const TileRight_ &rhs,
                      const TileRightScale_ &) {
  MATMACC(dst, lhs, rhs);
}

template <typename DstTile, typename SrcTile>
inline void TCVT(DstTile &dst, const SrcTile &src) {
  dst.raw() = linx::detail::teplUnary<tepl::TCVT, detail::tileSizeCode<DstTile>(),
                                      detail::tileDTypeCode<DstTile>()>(
      src.raw());
}

template <typename DstTile, typename SrcTile>
inline void TCVT_DN2NZ(DstTile &dst, const SrcTile &src) {
  TCVT(dst, src);
}

template <typename DstTile, typename SrcTile>
inline void TMOV_DN2NZ(DstTile &dst, const SrcTile &src) {
  TCVT(dst, src);
}

template <typename DstTile, typename SrcTile>
inline void TMOV_NORM(DstTile &dst, const SrcTile &src) {
  TCVT(dst, src);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TADD(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.raw() = linx::detail::teplBinary<tepl::TADD, detail::tileSizeCode<DstTile>(),
                                       detail::tileDTypeCode<DstTile>()>(
      src0.raw(), src1.raw());
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TADDS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.raw() = linx::detail::teplBinaryScalar<tepl::TADDS,
                                             detail::tileSizeCode<DstTile>(),
                                             detail::tileDTypeCode<DstTile>(), 1u>(
      src.raw(), scalar);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TSUB(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.raw() = linx::detail::teplBinary<tepl::TSUB, detail::tileSizeCode<DstTile>(),
                                       detail::tileDTypeCode<DstTile>()>(
      src0.raw(), src1.raw());
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TSUBS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.raw() = linx::detail::teplBinaryScalar<tepl::TSUBS,
                                             detail::tileSizeCode<DstTile>(),
                                             detail::tileDTypeCode<DstTile>(), 1u>(
      src.raw(), scalar);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TMUL(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.raw() = linx::detail::teplBinary<tepl::TMUL, detail::tileSizeCode<DstTile>(),
                                       detail::tileDTypeCode<DstTile>()>(
      src0.raw(), src1.raw());
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TDIV(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.raw() = linx::detail::teplBinary<tepl::TDIV, detail::tileSizeCode<DstTile>(),
                                       detail::tileDTypeCode<DstTile>()>(
      src0.raw(), src1.raw());
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TMAX(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.raw() = linx::detail::teplBinary<tepl::TMAX, detail::tileSizeCode<DstTile>(),
                                       detail::tileDTypeCode<DstTile>()>(
      src0.raw(), src1.raw());
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TMIN(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.raw() = linx::detail::teplBinary<tepl::TMIN, detail::tileSizeCode<DstTile>(),
                                       detail::tileDTypeCode<DstTile>()>(
      src0.raw(), src1.raw());
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TAND(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.raw() = linx::detail::teplBinary<tepl::TAND, detail::tileSizeCode<DstTile>(),
                                       detail::tileDTypeCode<DstTile>()>(
      src0.raw(), src1.raw());
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TANDS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.raw() = linx::detail::teplBinaryScalar<tepl::TANDS,
                                             detail::tileSizeCode<DstTile>(),
                                             detail::tileDTypeCode<DstTile>(), 1u>(
      src.raw(), scalar);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TOR(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.raw() = linx::detail::teplBinary<tepl::TOR, detail::tileSizeCode<DstTile>(),
                                       detail::tileDTypeCode<DstTile>()>(
      src0.raw(), src1.raw());
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TORS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.raw() = linx::detail::teplBinaryScalar<tepl::TORS,
                                             detail::tileSizeCode<DstTile>(),
                                             detail::tileDTypeCode<DstTile>(), 1u>(
      src.raw(), scalar);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TXOR(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.raw() = linx::detail::teplBinary<tepl::TXOR, detail::tileSizeCode<DstTile>(),
                                       detail::tileDTypeCode<DstTile>()>(
      src0.raw(), src1.raw());
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TXORS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.raw() = linx::detail::teplBinaryScalar<tepl::TXORS,
                                             detail::tileSizeCode<DstTile>(),
                                             detail::tileDTypeCode<DstTile>(), 1u>(
      src.raw(), scalar);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TSHL(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.raw() = linx::detail::teplBinary<tepl::TSHL, detail::tileSizeCode<DstTile>(),
                                       detail::tileDTypeCode<DstTile>()>(
      src0.raw(), src1.raw());
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TSHLS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.raw() = linx::detail::teplBinaryScalar<tepl::TSHLS,
                                             detail::tileSizeCode<DstTile>(),
                                             detail::tileDTypeCode<DstTile>(), 1u>(
      src.raw(), scalar);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TSHR(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.raw() = linx::detail::teplBinary<tepl::TSHR, detail::tileSizeCode<DstTile>(),
                                       detail::tileDTypeCode<DstTile>()>(
      src0.raw(), src1.raw());
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TSHRS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.raw() = linx::detail::teplBinaryScalar<tepl::TSHRS,
                                             detail::tileSizeCode<DstTile>(),
                                             detail::tileDTypeCode<DstTile>(), 1u>(
      src.raw(), scalar);
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TSLL(DstTile &dst, const SrcTile &src, Scalar scalar) {
  TSHLS(dst, src, scalar);
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TSRL(DstTile &dst, const SrcTile &src, Scalar scalar) {
  TSHRS(dst, src, scalar);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TCMP(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1,
                 CmpMode = CmpMode::EQ) {
  dst.raw() = linx::detail::teplBinary<tepl::TCMP, detail::tileSizeCode<DstTile>(),
                                       detail::tileDTypeCode<DstTile>()>(
      src0.raw(), src1.raw());
}

template <typename DstTile, typename MaskTile, typename SrcTile0,
          typename SrcTile1>
inline void TSEL(DstTile &dst, const MaskTile &mask, const SrcTile0 &src0,
                 const SrcTile1 &src1) {
  dst.raw() = linx::detail::teplTernary<tepl::TSEL,
                                        detail::tileSizeCode<DstTile>(),
                                        detail::tileDTypeCode<DstTile>()>(
      mask.raw(), src0.raw(), src1.raw());
}

template <typename DstTile, typename MaskTile, typename SrcTile0,
          typename SrcTile1>
inline void TSELECT(DstTile &dst, const MaskTile &mask, const SrcTile0 &src0,
                    const SrcTile1 &src1) {
  TSEL(dst, mask, src0, src1);
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TMAXS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.raw() = linx::detail::teplBinaryScalar<tepl::TMAXS,
                                             detail::tileSizeCode<DstTile>(),
                                             detail::tileDTypeCode<DstTile>(), 1u>(
      src.raw(), scalar);
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TMINS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.raw() = linx::detail::teplBinaryScalar<tepl::TMINS,
                                             detail::tileSizeCode<DstTile>(),
                                             detail::tileDTypeCode<DstTile>(), 1u>(
      src.raw(), scalar);
}

template <typename DstTile, typename SrcTile>
inline void TROWMAX(DstTile &dst, const SrcTile &src) {
  dst.raw() = linx::detail::teplUnary<tepl::TROWMAX,
                                      detail::tileSizeCode<DstTile>(),
                                      detail::tileDTypeCode<DstTile>()>(src.raw());
}

template <typename DstTile, typename SrcTile, typename TmpTile>
inline void TROWMAX(DstTile &dst, const SrcTile &src, TmpTile &) {
  TROWMAX(dst, src);
}

template <typename DstTile, typename SrcTile>
inline void TCOLMAX(DstTile &dst, const SrcTile &src) {
  dst.raw() = linx::detail::teplUnary<tepl::TCOLMAX,
                                      detail::tileSizeCode<DstTile>(),
                                      detail::tileDTypeCode<DstTile>()>(src.raw());
}

template <typename DstTile, typename SrcTile, typename TmpTile>
inline void TCOLMAX(DstTile &dst, const SrcTile &src, TmpTile &) {
  TCOLMAX(dst, src);
}

template <typename DstTile, typename SrcTile>
inline void TROWSUM(DstTile &dst, const SrcTile &src) {
  dst.raw() = linx::detail::teplUnary<tepl::TROWSUM,
                                      detail::tileSizeCode<DstTile>(),
                                      detail::tileDTypeCode<DstTile>()>(src.raw());
}

template <typename DstTile, typename SrcTile, typename TmpTile>
inline void TROWSUM(DstTile &dst, const SrcTile &src, TmpTile &) {
  TROWSUM(dst, src);
}

template <typename DstTile, typename SrcTile>
inline void TCOLSUM(DstTile &dst, const SrcTile &src) {
  dst.raw() = linx::detail::teplUnary<tepl::TCOLSUM,
                                      detail::tileSizeCode<DstTile>(),
                                      detail::tileDTypeCode<DstTile>()>(src.raw());
}

template <typename DstTile, typename SrcTile, typename TmpTile>
inline void TCOLSUM(DstTile &dst, const SrcTile &src, TmpTile &, bool = false) {
  TCOLSUM(dst, src);
}

template <typename DstTile, typename SrcTile>
inline void TROWPROD(DstTile &dst, const SrcTile &src) {
  dst.raw() = linx::detail::teplUnary<tepl::TROWPROD,
                                      detail::tileSizeCode<DstTile>(),
                                      detail::tileDTypeCode<DstTile>()>(src.raw());
}

template <typename DstTile, typename SrcTile, typename TmpTile>
inline void TROWPROD(DstTile &dst, const SrcTile &src, TmpTile &) {
  TROWPROD(dst, src);
}

template <typename DstTile, typename SrcTile>
inline void TCOLPROD(DstTile &dst, const SrcTile &src) {
  dst.raw() = linx::detail::teplUnary<tepl::TCOLPROD,
                                      detail::tileSizeCode<DstTile>(),
                                      detail::tileDTypeCode<DstTile>()>(src.raw());
}

template <typename DstTile, typename SrcTile, typename TmpTile>
inline void TCOLPROD(DstTile &dst, const SrcTile &src, TmpTile &, bool = false) {
  TCOLPROD(dst, src);
}

template <typename DstTile, typename SrcTile>
inline void TEXP(DstTile &dst, const SrcTile &src) {
  dst.raw() = linx::detail::teplUnary<tepl::TEXP, detail::tileSizeCode<DstTile>(),
                                      detail::tileDTypeCode<DstTile>()>(
      src.raw());
}

template <typename DstTile, typename SrcTile>
inline void TRECIP(DstTile &dst, const SrcTile &src) {
  dst.raw() = linx::detail::teplUnary<tepl::TRECIP,
                                      detail::tileSizeCode<DstTile>(),
                                      detail::tileDTypeCode<DstTile>()>(src.raw());
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TMULS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.raw() = linx::detail::teplBinaryScalar<tepl::TMULS,
                                             detail::tileSizeCode<DstTile>(),
                                             detail::tileDTypeCode<DstTile>(), 1u>(
      src.raw(), scalar);
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TDIVS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.raw() = linx::detail::teplBinaryScalar<tepl::TDIVS,
                                             detail::tileSizeCode<DstTile>(),
                                             detail::tileDTypeCode<DstTile>(), 1u>(
      src.raw(), scalar);
}

template <typename DstTile, typename SrcTile, typename Scalar>
inline void TREMS(DstTile &dst, const SrcTile &src, Scalar scalar) {
  dst.raw() = linx::detail::teplBinaryScalar<tepl::TREMS,
                                             detail::tileSizeCode<DstTile>(),
                                             detail::tileDTypeCode<DstTile>(), 1u>(
      src.raw(), scalar);
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
inline void TREM(DstTile &dst, const SrcTile0 &src0, const SrcTile1 &src1) {
  dst.raw() = linx::detail::teplBinary<tepl::TREM, detail::tileSizeCode<DstTile>(),
                                       detail::tileDTypeCode<DstTile>()>(
      src0.raw(), src1.raw());
}

template <typename DstTile, typename SrcTile, typename Scalar, typename TmpTile>
inline void TREMS(DstTile &dst, const SrcTile &src, Scalar scalar, TmpTile &) {
  TREMS(dst, src, scalar);
}

template <typename DstTile, typename Scalar>
inline void TEXPANDS(DstTile &dst, Scalar scalar) {
  dst.raw() = linx::detail::teplSplat<tepl::TEXPANDS,
                                      detail::tileSizeCode<DstTile>(),
                                      detail::tileDTypeCode<DstTile>(), 2u>(scalar);
}

template <typename DstTile, typename Scalar>
inline void TEXPANDSCALAR(DstTile &dst, Scalar scalar) {
  TEXPANDS(dst, scalar);
}

template <typename DstTile, typename SrcTile>
inline void TCAST(DstTile &dst, const SrcTile &src) {
  TCVT(dst, src);
}

template <typename DstTile, typename SrcTile>
inline void TCOPY(DstTile &dst, const SrcTile &src) {
  TMOV(dst, src);
}

template <typename TileData, typename T, int descending>
inline void TCI(TileData &dst, T start) {
  static_assert(descending == 0,
                "LinxISA v0.57: TCI descending form is not encoded yet");
  dst.raw() = linx::detail::teplSplat<tepl::TCI,
                                      detail::tileSizeCode<TileData>(),
                                      detail::tileDTypeCode<TileData>(), 2u>(start);
}

template <typename DstTile, typename SrcTile>
inline void TCOLEXPAND(DstTile &dst, const SrcTile &src) {
  dst.raw() = linx::detail::teplUnary<tepl::TCOLEXPAND,
                                      detail::tileSizeCode<DstTile>(),
                                      detail::tileDTypeCode<DstTile>()>(
      src.raw());
}

template <typename DstTile, typename SrcTile>
inline void TROWEXPAND(DstTile &dst, const SrcTile &src) {
  dst.raw() = linx::detail::teplUnary<tepl::TROWEXPAND,
                                      detail::tileSizeCode<DstTile>(),
                                      detail::tileDTypeCode<DstTile>()>(
      src.raw());
}

template <typename DstTile, typename SrcTile>
inline void TEXPANDCOL(DstTile &dst, const SrcTile &src) {
  TCOLEXPAND(dst, src);
}

} // namespace pto

#ifndef PTO_NO_GLOBAL_HALF_ALIAS
using __half = pto::fp16_t;
using __fp32 = float;
using pto_bf16_t = pto::bf16_t;
using pto_fp8_e4m3_t = pto::fp8_e4m3_t;
using pto_fp4_e2m1_t = pto::fp4_e2m1_t;
using __fp8_e4m3 = pto::fp8_e4m3_t;
using __fp4_e2m1 = pto::fp4_e2m1_t;
using __fp4_e2m1x2 = pto::fp4_e2m1_t;
using __fp4_hif4x2 = pto::fp4_e2m1_t;
#endif

#endif // PTO_COMMON_PTO_TILEOP_HPP
