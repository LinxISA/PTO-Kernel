#ifndef PTO_LINX_IMPL_BACKEND_HPP
#define PTO_LINX_IMPL_BACKEND_HPP

#include <stdint.h>
#if defined(PTO_HOST_SIM)
#include <math.h>
#include <string.h>
#endif
#include <common/linx_lowp_types.hpp>

namespace pto {
namespace linx {
namespace detail {

template <typename... Ts>
struct dependent_false {
  static constexpr bool value = false;
};

template <typename A, typename B>
struct is_same {
  static constexpr bool value = false;
};

template <typename T>
struct is_same<T, T> {
  static constexpr bool value = true;
};

template <typename T>
struct is_arithmetic {
  static constexpr bool value = false;
};

template <>
struct is_arithmetic<int> {
  static constexpr bool value = true;
};

template <>
struct is_arithmetic<unsigned> {
  static constexpr bool value = true;
};

template <>
struct is_arithmetic<short> {
  static constexpr bool value = true;
};

template <>
struct is_arithmetic<unsigned short> {
  static constexpr bool value = true;
};

template <>
struct is_arithmetic<signed char> {
  static constexpr bool value = true;
};

template <>
struct is_arithmetic<unsigned char> {
  static constexpr bool value = true;
};

template <>
struct is_arithmetic<long> {
  static constexpr bool value = true;
};

template <>
struct is_arithmetic<unsigned long> {
  static constexpr bool value = true;
};

template <>
struct is_arithmetic<long long> {
  static constexpr bool value = true;
};

template <>
struct is_arithmetic<unsigned long long> {
  static constexpr bool value = true;
};

template <>
struct is_arithmetic<float> {
  static constexpr bool value = true;
};

template <>
struct is_arithmetic<double> {
  static constexpr bool value = true;
};

template <>
struct is_arithmetic<pto::fp16_t> {
  static constexpr bool value = true;
};

template <>
struct is_arithmetic<pto::bf16_t> {
  static constexpr bool value = true;
};

template <>
struct is_arithmetic<pto::fp8_e4m3_t> {
  static constexpr bool value = true;
};

template <>
struct is_arithmetic<pto::fp4_e2m1_t> {
  static constexpr bool value = true;
};

template <typename T>
struct is_floating_point {
  static constexpr bool value = false;
};

template <>
struct is_floating_point<float> {
  static constexpr bool value = true;
};

template <>
struct is_floating_point<double> {
  static constexpr bool value = true;
};

template <>
struct is_floating_point<pto::bf16_t> {
  static constexpr bool value = true;
};

template <typename T>
struct DTypeCode {
  static_assert(dependent_false<T>::value,
                "LinxISA v0.57: unsupported tile dtype");
};

template <>
struct DTypeCode<int> { static constexpr unsigned value = 17u; };

template <>
struct DTypeCode<unsigned> { static constexpr unsigned value = 25u; };

template <>
struct DTypeCode<float> { static constexpr unsigned value = 1u; };

template <>
struct DTypeCode<signed char> { static constexpr unsigned value = 19u; };

template <>
struct DTypeCode<unsigned char> { static constexpr unsigned value = 27u; };

template <>
struct DTypeCode<short> { static constexpr unsigned value = 18u; };

template <>
struct DTypeCode<unsigned short> { static constexpr unsigned value = 26u; };

template <>
struct DTypeCode<long> { static constexpr unsigned value = 16u; };

template <>
struct DTypeCode<unsigned long> { static constexpr unsigned value = 24u; };

template <>
struct DTypeCode<long long> { static constexpr unsigned value = 16u; };

template <>
struct DTypeCode<unsigned long long> { static constexpr unsigned value = 24u; };

template <>
struct DTypeCode<double> { static constexpr unsigned value = 0u; };

template <>
struct DTypeCode<pto::fp16_t> { static constexpr unsigned value = 2u; };

template <>
struct DTypeCode<pto::bf16_t> { static constexpr unsigned value = 6u; };

template <>
struct DTypeCode<pto::fp8_e4m3_t> { static constexpr unsigned value = 3u; };

template <>
struct DTypeCode<pto::fp4_e2m1_t> { static constexpr unsigned value = 11u; };

constexpr unsigned kMinTileBytes = 512u;
constexpr unsigned kMaxTileBytes = 4096u;
constexpr unsigned kTileWords = kMaxTileBytes / sizeof(uint32_t);

#if defined(PTO_HOST_SIM)
struct RawTile {
  alignas(64) uint32_t words[kTileWords];
};
#else
using RawTile = int __attribute__((__vector_size__(4096), __aligned__(64)));
#endif

constexpr unsigned clampTileBytes(unsigned bytes) {
  return bytes < kMinTileBytes ? kMinTileBytes
                               : (bytes > kMaxTileBytes ? kMaxTileBytes : bytes);
}

constexpr unsigned nextPow2(unsigned value) {
  unsigned p = 1u;
  while (p < value && p < kMaxTileBytes)
    p <<= 1u;
  return p;
}

constexpr unsigned sizeCodeFromBytes(unsigned bytes) {
  const unsigned clipped = clampTileBytes(bytes);
  const unsigned p2 = nextPow2(clipped);
  unsigned code = 0u;
  while ((1u << (code + 4u)) < p2)
    ++code;
  if (code < 5u)
    code = 5u;
  if (code > 8u)
    code = 8u;
  return code;
}

constexpr unsigned dtypeElemBits(unsigned dtype) {
  switch (dtype & 0x1fu) {
  case 0u:  // FP64
  case 16u: // INT64
  case 24u: // UINT64
    return 64u;
  case 1u:  // FP32
  case 17u: // INT32
  case 25u: // UINT32
    return 32u;
  case 2u:  // FP16
  case 6u:  // BF16
  case 18u: // INT16
  case 26u: // UINT16
    return 16u;
  case 3u:  // FP8
  case 7u:  // FPL8
  case 19u: // INT8
  case 27u: // UINT8
    return 8u;
  case 11u: // FP4
  case 12u: // FPL4
  case 20u: // INT4
  case 28u: // UINT4
    return 4u;
  default:
    return 32u;
  }
}

constexpr unsigned dtypeElemBytesForStorage(unsigned dtype) {
  const unsigned bits = dtypeElemBits(dtype);
  return (bits + 7u) / 8u;
}

constexpr unsigned dtypeElemCountForBytes(uint64_t bytes, unsigned dtype) {
  const unsigned bits = dtypeElemBits(dtype);
  if (bits == 0u)
    return 0u;
  const uint64_t total_bits = bytes * 8u;
  return static_cast<unsigned>(total_bits / bits);
}

template <typename Scalar>
inline long long encodeScalar(Scalar value) {
  static_assert(is_arithmetic<Scalar>::value,
                "LinxISA v0.57: scalar operand must be arithmetic");
  if constexpr (is_same<Scalar, pto::fp16_t>::value) {
    return static_cast<long long>(value.bits);
  } else if constexpr (is_same<Scalar, pto::bf16_t>::value) {
    return static_cast<long long>(value.bits);
  } else if constexpr (is_same<Scalar, pto::fp8_e4m3_t>::value) {
    return static_cast<long long>(value.bits);
  } else if constexpr (is_same<Scalar, pto::fp4_e2m1_t>::value) {
    return static_cast<long long>(value.bits & 0x0fu);
  } else if constexpr (is_floating_point<Scalar>::value) {
    if constexpr (sizeof(Scalar) == sizeof(uint32_t)) {
      union {
        Scalar f;
        uint32_t u;
      } cvt = {value};
      return static_cast<long long>(cvt.u);
    } else if constexpr (sizeof(Scalar) == sizeof(uint64_t)) {
      union {
        Scalar f;
        uint64_t u;
      } cvt = {value};
      return static_cast<long long>(cvt.u);
    } else {
      return static_cast<long long>(value);
    }
  } else {
    return static_cast<long long>(value);
  }
}

#if defined(PTO_HOST_SIM)

inline uint64_t sizeBytesFromCode(unsigned size_code) {
  return (size_code < 60u) ? (1ull << (size_code + 4u)) : 0ull;
}

template <typename T>
inline uint32_t bitCastToU32(T value) {
  static_assert(sizeof(T) == sizeof(uint32_t), "bitCastToU32 requires 32-bit type");
  uint32_t out = 0;
  memcpy(&out, &value, sizeof(uint32_t));
  return out;
}

template <typename T>
inline T bitCastFromU32(uint32_t bits) {
  static_assert(sizeof(T) == sizeof(uint32_t), "bitCastFromU32 requires 32-bit type");
  T out{};
  memcpy(&out, &bits, sizeof(uint32_t));
  return out;
}

inline float scalarAsF32(long long scalar_bits) {
  uint32_t bits = static_cast<uint32_t>(scalar_bits & 0xffffffffull);
  return bitCastFromU32<float>(bits);
}

inline int32_t scalarAsI32(long long scalar_bits) {
  return static_cast<int32_t>(scalar_bits & 0xffffffffull);
}

inline uint32_t scalarToWordDType(long long scalar_bits, unsigned dtype) {
  switch (dtype & 0x1fu) {
  case 2u:
  case 6u:
    return static_cast<uint32_t>(static_cast<uint16_t>(scalar_bits & 0xffffu));
  case 3u:
    return static_cast<uint32_t>(static_cast<uint8_t>(scalar_bits & 0xffu));
  case 11u:
    return static_cast<uint32_t>(static_cast<uint8_t>(scalar_bits & 0x0fu));
  default:
    return static_cast<uint32_t>(scalar_bits & 0xffffffffu);
  }
}

inline uint32_t quantizeF32ToWord(float x, unsigned dtype) {
  switch (dtype & 0x1fu) {
  case 2u:
    return pto::lowp_word_from_fp16(pto::float_to_fp16(x));
  case 6u:
    return pto::lowp_word_from_bf16(pto::float_to_bf16(x));
  case 3u:
    return pto::lowp_word_from_fp8(pto::float_to_fp8_e4m3(x));
  case 11u:
    return pto::lowp_word_from_fp4(pto::float_to_fp4_e2m1(x));
  default:
    return bitCastToU32<float>(x);
  }
}

inline float dequantWordToF32(uint32_t word, unsigned dtype) {
  switch (dtype & 0x1fu) {
  case 2u:
    return pto::fp16_to_float(pto::fp16_from_lowp_word(word));
  case 6u:
    return pto::bf16_to_float(pto::bf16_from_lowp_word(word));
  case 3u:
    return pto::fp8_e4m3_to_float(pto::fp8_from_lowp_word(word));
  case 11u:
    return pto::fp4_e2m1_to_float(pto::fp4_from_lowp_word(word));
  default:
    return bitCastFromU32<float>(word);
  }
}

template <unsigned TileOpcode, unsigned DType>
inline RawTile teplUnaryHost(const RawTile &src, unsigned elems) {
  RawTile out{};
  for (unsigned i = 0; i < kTileWords; ++i)
    out.words[i] = 0u;

  switch (TileOpcode & 0x3ffu) {
  case 0x019u: // TCVT
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float f = dequantWordToF32(src.words[i], DType);
      out.words[i] = quantizeF32ToWord(f, DType);
    }
    break;
  case 0x015u: // TCOLMAX (fallback: identity under host backend)
  case 0x017u: // TCOLPROD (fallback: identity under host backend)
  case 0x018u: // TCOLSUM (fallback: identity under host backend)
  case 0x00du: // TCOLEXPAND (fallback: identity under host backend)
  case 0x047u: // TROWMAX (fallback: identity under host backend)
  case 0x049u: // TROWPROD (fallback: identity under host backend)
  case 0x04au: // TROWSUM (fallback: identity under host backend)
  case 0x03fu: // TROWEXPAND (fallback: identity under host backend)
    for (unsigned i = 0; i < elems && i < kTileWords; ++i)
      out.words[i] = src.words[i];
    break;
  case 0x01cu: // TEXP
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float f = dequantWordToF32(src.words[i], DType);
      out.words[i] = quantizeF32ToWord(expf(f), DType);
    }
    break;
  case 0x039u: // TRECIP
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float f = dequantWordToF32(src.words[i], DType);
      float inv = (f == 0.0f) ? 0.0f : (1.0f / f);
      out.words[i] = quantizeF32ToWord(inv, DType);
    }
    break;
  default:
    // Unsupported op in host backend: keep destination zeroed.
    break;
  }
  return out;
}

template <unsigned TileOpcode, unsigned DType>
inline RawTile teplBinaryHost(const RawTile &lhs, const RawTile &rhs, unsigned elems) {
  RawTile out{};
  for (unsigned i = 0; i < kTileWords; ++i)
    out.words[i] = 0u;

  switch (TileOpcode & 0x3ffu) {
  case 0x001u: // TADD
  case 0x003u: // TADDS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float a = dequantWordToF32(lhs.words[i], DType);
      const float b = dequantWordToF32(rhs.words[i], DType);
      out.words[i] = quantizeF32ToWord(a + b, DType);
    }
    break;
  case 0x055u: // TSUB
  case 0x057u: // TSUBS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float a = dequantWordToF32(lhs.words[i], DType);
      const float b = dequantWordToF32(rhs.words[i], DType);
      out.words[i] = quantizeF32ToWord(a - b, DType);
    }
    break;
  case 0x02au: // TMUL
  case 0x02bu: { // TMULS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float a = dequantWordToF32(lhs.words[i], DType);
      const float b = dequantWordToF32(rhs.words[i], DType);
      out.words[i] = quantizeF32ToWord(a * b, DType);
    }
    break;
  }
  case 0x01au: // TDIV
  case 0x01bu: { // TDIVS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float a = dequantWordToF32(lhs.words[i], DType);
      const float b = dequantWordToF32(rhs.words[i], DType);
      const float q = (b == 0.0f) ? 0.0f : (a / b);
      out.words[i] = quantizeF32ToWord(q, DType);
    }
    break;
  }
  case 0x025u: // TMAX
  case 0x026u: { // TMAXS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float a = dequantWordToF32(lhs.words[i], DType);
      const float b = dequantWordToF32(rhs.words[i], DType);
      out.words[i] = quantizeF32ToWord(a > b ? a : b, DType);
    }
    break;
  }
  case 0x027u: // TMIN
  case 0x028u: { // TMINS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const float a = dequantWordToF32(lhs.words[i], DType);
      const float b = dequantWordToF32(rhs.words[i], DType);
      out.words[i] = quantizeF32ToWord(a < b ? a : b, DType);
    }
    break;
  }
  case 0x005u: // TAND
  case 0x006u: // TANDS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i)
      out.words[i] = lhs.words[i] & rhs.words[i];
    break;
  case 0x02eu: // TOR
  case 0x02fu: // TORS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i)
      out.words[i] = lhs.words[i] | rhs.words[i];
    break;
  case 0x05au: // TXOR
  case 0x05bu: // TXORS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i)
      out.words[i] = lhs.words[i] ^ rhs.words[i];
    break;
  case 0x04fu: // TSHL
  case 0x050u: // TSHLS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i)
      out.words[i] = lhs.words[i] << (rhs.words[i] & 31u);
    break;
  case 0x051u: // TSHR
  case 0x052u: // TSHRS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i)
      out.words[i] = lhs.words[i] >> (rhs.words[i] & 31u);
    break;
  case 0x03cu: { // TREMS
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const int32_t a = static_cast<int32_t>(lhs.words[i]);
      const int32_t b = static_cast<int32_t>(rhs.words[i]);
      out.words[i] =
          (b == 0 || (a == INT32_MIN && b == -1)) ? 0u : static_cast<uint32_t>(a % b);
    }
    break;
  }
  case 0x03bu: { // TREM
    for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
      const int32_t a = static_cast<int32_t>(lhs.words[i]);
      const int32_t b = static_cast<int32_t>(rhs.words[i]);
      out.words[i] =
          (b == 0 || (a == INT32_MIN && b == -1)) ? 0u : static_cast<uint32_t>(a % b);
    }
    break;
  }
  case 0x009u: { // TCMP, fallback EQ mask
    for (unsigned i = 0; i < elems && i < kTileWords; ++i)
      out.words[i] = lhs.words[i] == rhs.words[i] ? 0xffffffffu : 0u;
    break;
  }
  default:
    break;
  }
  return out;
}

template <unsigned TileOpcode, unsigned DType>
inline RawTile teplTernaryHost(const RawTile &sel, const RawTile &lhs,
                               const RawTile &rhs, unsigned elems) {
  RawTile out{};
  for (unsigned i = 0; i < kTileWords; ++i)
    out.words[i] = 0u;

  switch (TileOpcode & 0x3ffu) {
  case 0x04du: // TSEL
    for (unsigned i = 0; i < elems && i < kTileWords; ++i)
      out.words[i] = sel.words[i] ? lhs.words[i] : rhs.words[i];
    break;
  default:
    break;
  }
  (void)DType;
  return out;
}

#endif

template <unsigned SizeCode, unsigned DType, long long Layout, long long LB0,
          long long LB1, long long StrideBytes>
inline RawTile tileTLoad(const void *base) {
  static_assert(SizeCode >= 5u && SizeCode <= 8u,
                "LinxISA v0.57: size_code must be in [5,8]");
#if defined(PTO_HOST_SIM)
  (void)Layout;
  RawTile out{};
  for (unsigned i = 0; i < kTileWords; ++i)
    out.words[i] = 0u;

  const uint64_t bytes64 = sizeBytesFromCode(SizeCode);
  const unsigned elem_bytes = dtypeElemBytesForStorage(DType);
  const unsigned elem_bits = dtypeElemBits(DType);
  if (bytes64 == 0 || bytes64 > kMaxTileBytes || elem_bits == 0u ||
      (bytes64 % elem_bytes) != 0u)
    return out;

  const unsigned max_elems = dtypeElemCountForBytes(bytes64, DType);
  const uint64_t cols = (LB0 > 0) ? static_cast<uint64_t>(LB0)
                                  : static_cast<uint64_t>(max_elems);
  const uint64_t rows = (LB1 > 0) ? static_cast<uint64_t>(LB1) : 1u;
  if (rows == 0u || cols == 0u)
    return out;
  if (rows > (UINT64_MAX / cols))
    return out;
  const uint64_t active = rows * cols;
  if (active > max_elems)
    return out;

  const uint64_t row_span_bits = cols * elem_bits;
  const uint64_t row_span_bytes = (row_span_bits + 7u) / 8u;
  const uint64_t stride_bytes =
      (StrideBytes > 0) ? static_cast<uint64_t>(StrideBytes) : row_span_bytes;
  if (stride_bytes < row_span_bytes ||
      (elem_bytes != 0u && (stride_bytes % elem_bytes) != 0u)) {
    return out;
  }

  const uint8_t *src = reinterpret_cast<const uint8_t *>(base);
  for (uint64_t r = 0; r < rows; ++r) {
    const uint64_t row_base = r * stride_bytes;
    for (uint64_t c = 0; c < cols; ++c) {
      const uint64_t idx64 = r * cols + c;
      if (idx64 >= kTileWords)
        return out;
      const unsigned idx = static_cast<unsigned>(idx64);

      uint32_t value = 0u;
      if (elem_bits == 4u) {
        const uint64_t byte_addr = row_base + (c >> 1u);
        const uint8_t packed = src[byte_addr];
        value = ((c & 1u) == 0u) ? (packed & 0x0fu) : ((packed >> 4u) & 0x0fu);
      } else if (elem_bytes == 1u) {
        value = static_cast<uint32_t>(src[row_base + c]);
      } else if (elem_bytes == 2u) {
        uint16_t v = 0u;
        memcpy(&v, src + row_base + c * 2u, sizeof(v));
        value = static_cast<uint32_t>(v);
      } else if (elem_bytes == 4u) {
        uint32_t v = 0u;
        memcpy(&v, src + row_base + c * 4u, sizeof(v));
        value = v;
      } else if (elem_bytes == 8u) {
        uint64_t v = 0u;
        memcpy(&v, src + row_base + c * 8u, sizeof(v));
        value = static_cast<uint32_t>(v & 0xffffffffu);
      } else {
        return out;
      }
      out.words[idx] = value;
    }
  }
  return out;
#else
  return __builtin_linx_tile_tload(base, SizeCode, DType, Layout, LB0, LB1,
                                   StrideBytes);
#endif
}

template <unsigned SizeCode, unsigned DType, long long Layout, long long LB0,
          long long LB1, long long StrideBytes>
inline RawTile tileMGather(const void *base, RawTile index) {
  static_assert(SizeCode >= 5u && SizeCode <= 8u,
                "LinxISA v0.57: size_code must be in [5,8]");
#if defined(PTO_HOST_SIM)
  (void)Layout;
  const uint64_t bytes64 = sizeBytesFromCode(SizeCode);
  const unsigned elem_bytes = dtypeElemBytesForStorage(DType);
  const unsigned elems =
      (bytes64 == 0 || bytes64 > kMaxTileBytes || elem_bytes == 0)
          ? 0u
          : dtypeElemCountForBytes(bytes64, DType);
  const uint8_t *src = reinterpret_cast<const uint8_t *>(base);
  RawTile out{};
  memset(out.words, 0, sizeof(out.words));
  for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
    uint32_t byte_offset = index.words[i];
    memcpy(reinterpret_cast<uint8_t *>(out.words) + i * elem_bytes,
           src + byte_offset, elem_bytes);
  }
  return out;
#else
  return __builtin_linx_tile_mgather(base, index, SizeCode, DType, Layout, LB0,
                                     LB1, StrideBytes);
#endif
}

template <unsigned SizeCode, unsigned DType, long long Layout, long long LB0,
          long long LB1, long long StrideBytes>
inline void tileTStore(void *base, RawTile tile) {
  static_assert(SizeCode >= 5u && SizeCode <= 8u,
                "LinxISA v0.57: size_code must be in [5,8]");
#if defined(PTO_HOST_SIM)
  (void)Layout;
  const uint64_t bytes64 = sizeBytesFromCode(SizeCode);
  const unsigned elem_bytes = dtypeElemBytesForStorage(DType);
  const unsigned elem_bits = dtypeElemBits(DType);
  if (bytes64 == 0 || bytes64 > kMaxTileBytes || elem_bits == 0u ||
      (bytes64 % elem_bytes) != 0u)
    return;

  const unsigned max_elems = dtypeElemCountForBytes(bytes64, DType);
  const uint64_t cols = (LB0 > 0) ? static_cast<uint64_t>(LB0)
                                  : static_cast<uint64_t>(max_elems);
  const uint64_t rows = (LB1 > 0) ? static_cast<uint64_t>(LB1) : 1u;
  if (rows == 0u || cols == 0u)
    return;
  if (rows > (UINT64_MAX / cols))
    return;
  const uint64_t active = rows * cols;
  if (active > max_elems)
    return;

  const uint64_t row_span_bits = cols * elem_bits;
  const uint64_t row_span_bytes = (row_span_bits + 7u) / 8u;
  const uint64_t stride_bytes =
      (StrideBytes > 0) ? static_cast<uint64_t>(StrideBytes) : row_span_bytes;
  if (stride_bytes < row_span_bytes ||
      (elem_bytes != 0u && (stride_bytes % elem_bytes) != 0u)) {
    return;
  }

  uint8_t *dst = reinterpret_cast<uint8_t *>(base);
  for (uint64_t r = 0; r < rows; ++r) {
    const uint64_t row_base = r * stride_bytes;
    for (uint64_t c = 0; c < cols; ++c) {
      const uint64_t idx64 = r * cols + c;
      if (idx64 >= kTileWords)
        return;
      const uint32_t value = tile.words[static_cast<unsigned>(idx64)];

      if (elem_bits == 4u) {
        const uint64_t byte_addr = row_base + (c >> 1u);
        uint8_t packed = dst[byte_addr];
        const uint8_t nibble = static_cast<uint8_t>(value & 0x0fu);
        if ((c & 1u) == 0u)
          packed = static_cast<uint8_t>((packed & 0xf0u) | nibble);
        else
          packed = static_cast<uint8_t>((packed & 0x0fu) | (nibble << 4u));
        dst[byte_addr] = packed;
      } else if (elem_bytes == 1u) {
        dst[row_base + c] = static_cast<uint8_t>(value & 0xffu);
      } else if (elem_bytes == 2u) {
        const uint16_t v = static_cast<uint16_t>(value & 0xffffu);
        memcpy(dst + row_base + c * 2u, &v, sizeof(v));
      } else if (elem_bytes == 4u) {
        memcpy(dst + row_base + c * 4u, &value, sizeof(value));
      } else if (elem_bytes == 8u) {
        const uint64_t v = static_cast<uint64_t>(value);
        memcpy(dst + row_base + c * 8u, &v, sizeof(v));
      } else {
        return;
      }
    }
  }
#else
  __builtin_linx_tile_tstore(base, tile, SizeCode, DType, Layout, LB0, LB1,
                             StrideBytes);
#endif
}

template <unsigned SizeCode, unsigned DType, long long Layout, long long LB0,
          long long LB1, long long StrideBytes>
inline void tileMScatter(void *base, RawTile tile, RawTile index) {
  static_assert(SizeCode >= 5u && SizeCode <= 8u,
                "LinxISA v0.57: size_code must be in [5,8]");
#if defined(PTO_HOST_SIM)
  (void)Layout;
  const uint64_t bytes64 = sizeBytesFromCode(SizeCode);
  const unsigned elem_bytes = dtypeElemBytesForStorage(DType);
  const unsigned elems =
      (bytes64 == 0 || bytes64 > kMaxTileBytes || elem_bytes == 0)
          ? 0u
          : dtypeElemCountForBytes(bytes64, DType);
  uint8_t *dst = reinterpret_cast<uint8_t *>(base);
  for (unsigned i = 0; i < elems && i < kTileWords; ++i) {
    uint32_t byte_offset = index.words[i];
    memcpy(dst + byte_offset,
           reinterpret_cast<const uint8_t *>(tile.words) + i * elem_bytes,
           elem_bytes);
  }
#else
  __builtin_linx_tile_mscatter(base, tile, index, SizeCode, DType, Layout, LB0,
                               LB1, StrideBytes);
#endif
}

template <unsigned M, unsigned N, unsigned K>
inline RawTile cubeMamulb(RawTile lhs, RawTile rhs) {
  static_assert(M <= 0xffu && N <= 0xffu && K <= 0xffu,
                "LinxISA v0.57: cube dimensions must fit u8");
#if defined(PTO_HOST_SIM)
  RawTile out{};
  for (unsigned i = 0; i < kTileWords; ++i)
    out.words[i] = 0u;

  for (unsigned i = 0; i < M; ++i) {
    for (unsigned j = 0; j < N; ++j) {
      int64_t acc = 0;
      for (unsigned k = 0; k < K; ++k) {
        const unsigned a_idx = i * K + k;
        const unsigned b_idx = k * N + j;
        if (a_idx >= kTileWords || b_idx >= kTileWords)
          continue;
        const int32_t a = static_cast<int32_t>(lhs.words[a_idx]);
        const int32_t b = static_cast<int32_t>(rhs.words[b_idx]);
        acc += static_cast<int64_t>(a) * static_cast<int64_t>(b);
      }
      const unsigned out_idx = i * N + j;
      if (out_idx < kTileWords)
        out.words[out_idx] = static_cast<uint32_t>(static_cast<int32_t>(acc));
    }
  }
  return out;
#else
  return __builtin_linx_cube_mamulb(lhs, rhs, M, N, K);
#endif
}

template <unsigned M, unsigned N, unsigned K>
inline RawTile cubeMamulbAcc(RawTile acc, RawTile lhs, RawTile rhs) {
  static_assert(M <= 0xffu && N <= 0xffu && K <= 0xffu,
                "LinxISA v0.57: cube dimensions must fit u8");
#if defined(PTO_HOST_SIM)
  RawTile out = acc;
  for (unsigned i = 0; i < M; ++i) {
    for (unsigned j = 0; j < N; ++j) {
      const unsigned out_idx = i * N + j;
      int64_t sum = (out_idx < kTileWords)
                        ? static_cast<int32_t>(out.words[out_idx])
                        : 0;
      for (unsigned k = 0; k < K; ++k) {
        const unsigned a_idx = i * K + k;
        const unsigned b_idx = k * N + j;
        if (a_idx >= kTileWords || b_idx >= kTileWords)
          continue;
        const int32_t a = static_cast<int32_t>(lhs.words[a_idx]);
        const int32_t b = static_cast<int32_t>(rhs.words[b_idx]);
        sum += static_cast<int64_t>(a) * static_cast<int64_t>(b);
      }
      if (out_idx < kTileWords)
        out.words[out_idx] = static_cast<uint32_t>(static_cast<int32_t>(sum));
    }
  }
  return out;
#else
  return __builtin_linx_cube_mamulb_acc(acc, lhs, rhs, M, N, K);
#endif
}

template <unsigned TileOpcode, unsigned SizeCode, unsigned DType>
inline RawTile teplUnary(RawTile src) {
  static_assert(TileOpcode <= 0x3ffu,
                "LinxISA v0.57: TEPL tile opcode must fit u10");
  static_assert(SizeCode >= 5u && SizeCode <= 8u,
                "LinxISA v0.57: size_code must be in [5,8]");
#if defined(PTO_HOST_SIM)
  const uint64_t bytes64 = sizeBytesFromCode(SizeCode);
  const unsigned elem_bytes = dtypeElemBytesForStorage(DType);
  const unsigned elems =
      (bytes64 == 0 || bytes64 > kMaxTileBytes || elem_bytes == 0)
          ? 0u
          : dtypeElemCountForBytes(bytes64, DType);
  return teplUnaryHost<TileOpcode, DType>(src, elems);
#else
  return __builtin_linx_tepl_unary(src, TileOpcode, SizeCode, DType);
#endif
}

template <unsigned TileOpcode, unsigned SizeCode, unsigned DType>
inline RawTile teplBinary(RawTile lhs, RawTile rhs) {
  static_assert(TileOpcode <= 0x3ffu,
                "LinxISA v0.57: TEPL tile opcode must fit u10");
  static_assert(SizeCode >= 5u && SizeCode <= 8u,
                "LinxISA v0.57: size_code must be in [5,8]");
#if defined(PTO_HOST_SIM)
  const uint64_t bytes64 = sizeBytesFromCode(SizeCode);
  const unsigned elem_bytes = dtypeElemBytesForStorage(DType);
  const unsigned elems =
      (bytes64 == 0 || bytes64 > kMaxTileBytes || elem_bytes == 0)
          ? 0u
          : dtypeElemCountForBytes(bytes64, DType);
  return teplBinaryHost<TileOpcode, DType>(lhs, rhs, elems);
#else
  return __builtin_linx_tepl_binary(lhs, rhs, TileOpcode, SizeCode, DType);
#endif
}

template <unsigned TileOpcode, unsigned SizeCode, unsigned DType>
inline RawTile teplTernary(RawTile src0, RawTile src1, RawTile src2) {
  static_assert(TileOpcode <= 0x3ffu,
                "LinxISA v0.57: TEPL tile opcode must fit u10");
  static_assert(SizeCode >= 5u && SizeCode <= 8u,
                "LinxISA v0.57: size_code must be in [5,8]");
#if defined(PTO_HOST_SIM)
  const uint64_t bytes64 = sizeBytesFromCode(SizeCode);
  const unsigned elem_bytes = dtypeElemBytesForStorage(DType);
  const unsigned elems =
      (bytes64 == 0 || bytes64 > kMaxTileBytes || elem_bytes == 0)
          ? 0u
          : dtypeElemCountForBytes(bytes64, DType);
  return teplTernaryHost<TileOpcode, DType>(src0, src1, src2, elems);
#else
  return __builtin_linx_tepl_ternary(src0, src1, src2, TileOpcode, SizeCode,
                                     DType);
#endif
}

template <unsigned TileOpcode, unsigned SizeCode, unsigned DType, unsigned Mode,
          typename Scalar>
inline RawTile teplBinaryScalar(RawTile lhs, Scalar scalar) {
  static_assert(TileOpcode <= 0x3ffu,
                "LinxISA v0.57: TEPL tile opcode must fit u10");
  static_assert(SizeCode >= 5u && SizeCode <= 8u,
                "LinxISA v0.57: size_code must be in [5,8]");
  static_assert(Mode == 1u,
                "LinxISA v0.57: tepl.binary.scalar requires operand mode=VS(1)");
#if defined(PTO_HOST_SIM)
  RawTile rhs{};
  const uint64_t bytes64 = sizeBytesFromCode(SizeCode);
  const unsigned elem_bytes = dtypeElemBytesForStorage(DType);
  const unsigned elems =
      (bytes64 == 0 || bytes64 > kMaxTileBytes || elem_bytes == 0)
          ? 0u
          : dtypeElemCountForBytes(bytes64, DType);
  const long long bits = encodeScalar(scalar);
  const uint32_t scalar_word = scalarToWordDType(bits, DType);
  for (unsigned i = 0; i < elems && i < kTileWords; ++i)
    rhs.words[i] = scalar_word;
  return teplBinaryHost<TileOpcode, DType>(lhs, rhs, elems);
#else
  return __builtin_linx_tepl_binary_scalar(lhs, encodeScalar(scalar), TileOpcode,
                                           SizeCode, DType, Mode);
#endif
}

template <unsigned TileOpcode, unsigned SizeCode, unsigned DType, unsigned Mode,
          typename Scalar>
inline RawTile teplSplat(Scalar scalar) {
  static_assert(TileOpcode <= 0x3ffu,
                "LinxISA v0.57: TEPL tile opcode must fit u10");
  static_assert(SizeCode >= 5u && SizeCode <= 8u,
                "LinxISA v0.57: size_code must be in [5,8]");
  static_assert(Mode == 2u,
                "LinxISA v0.57: tepl.splat requires operand mode=SV(2)");
#if defined(PTO_HOST_SIM)
  RawTile out{};
  for (unsigned i = 0; i < kTileWords; ++i)
    out.words[i] = 0u;

  const uint64_t bytes64 = sizeBytesFromCode(SizeCode);
  const unsigned elem_bytes = dtypeElemBytesForStorage(DType);
  const unsigned elems =
      (bytes64 == 0 || bytes64 > kMaxTileBytes || elem_bytes == 0)
          ? 0u
          : dtypeElemCountForBytes(bytes64, DType);

  if ((TileOpcode & 0x3ffu) != 0x008u &&
      (TileOpcode & 0x3ffu) != 0x01du)
    return out;

  const long long bits = encodeScalar(scalar);
  const uint32_t scalar_word = scalarToWordDType(bits, DType);
  for (unsigned i = 0; i < elems && i < kTileWords; ++i)
    out.words[i] = scalar_word;
  return out;
#else
  return __builtin_linx_tepl_splat(encodeScalar(scalar), TileOpcode, SizeCode,
                                   DType, Mode);
#endif
}

template <unsigned SizeCode, unsigned DType, long long Layout, unsigned HasLayout,
          unsigned Mode>
inline RawTile tileTMov(RawTile src) {
  static_assert(SizeCode >= 5u && SizeCode <= 8u,
                "LinxISA v0.57: size_code must be in [5,8]");
  static_assert(HasLayout <= 1u, "LinxISA v0.57: has_layout must be bool");
  static_assert(Mode <= 1u,
                "LinxISA v0.57: tmov mode must be 0(V2V) or 1(A2V)");
#if defined(PTO_HOST_SIM)
  (void)DType;
  (void)Layout;
  (void)HasLayout;
  (void)Mode;
  return src;
#else
  return __builtin_linx_tile_tmov(src, Mode, SizeCode, DType, Layout, HasLayout);
#endif
}

template <unsigned SizeCode, unsigned DType, unsigned DstRows,
          unsigned DstCols, unsigned SrcRows, unsigned SrcCols>
inline RawTile tileTInsert(RawTile dst, RawTile src, long long meta) {
  static_assert(SizeCode >= 5u && SizeCode <= 8u,
                "LinxISA v0.57: size_code must be in [5,8]");
  static_assert(DstRows > 0u && DstCols > 0u && SrcRows > 0u && SrcCols > 0u,
                "LinxISA v0.57: TINSERT tile dimensions must be positive");
  static_assert(DstRows <= 0xffffu && DstCols <= 0xffffu &&
                    SrcRows <= 0xffffu && SrcCols <= 0xffffu,
                "LinxISA v0.57: TINSERT dimensions must fit u16 metadata");
#if defined(PTO_HOST_SIM)
  RawTile out = dst;
  const unsigned index_row = static_cast<unsigned>((meta >> 32) & 0xffffffffu);
  const unsigned index_col = static_cast<unsigned>(meta & 0xffffffffu);
  for (unsigned r = 0; r < SrcRows; ++r) {
    const unsigned dst_r = index_row + r;
    if (dst_r >= DstRows)
      continue;
    for (unsigned c = 0; c < SrcCols; ++c) {
      const unsigned dst_c = index_col + c;
      if (dst_c >= DstCols)
        continue;
      const unsigned src_idx = r * SrcCols + c;
      const unsigned dst_idx = dst_r * DstCols + dst_c;
      if (src_idx < kTileWords && dst_idx < kTileWords)
        out.words[dst_idx] = src.words[src_idx];
    }
  }
  (void)DType;
  return out;
#else
  return __builtin_linx_tile_tinsert(dst, src, SizeCode, DType, DstRows,
                                    DstCols, SrcRows, SrcCols, meta);
#endif
}

template <unsigned SizeCode, unsigned DType, unsigned DstRows,
          unsigned DstCols, unsigned SrcRows, unsigned SrcCols>
inline RawTile tileTTrans(RawTile src, RawTile tmp) {
  static_assert(SizeCode >= 5u && SizeCode <= 8u,
                "LinxISA v0.57: size_code must be in [5,8]");
  static_assert(DstRows > 0u && DstCols > 0u && SrcRows > 0u && SrcCols > 0u,
                "LinxISA v0.57: TTRANS tile dimensions must be positive");
  static_assert(DstRows <= 0xffffu && DstCols <= 0xffffu &&
                    SrcRows <= 0xffffu && SrcCols <= 0xffffu,
                "LinxISA v0.57: TTRANS dimensions must fit u16 metadata");
#if defined(PTO_HOST_SIM)
  RawTile out{};
  for (unsigned i = 0; i < kTileWords; ++i)
    out.words[i] = 0u;
  for (unsigned r = 0; r < SrcRows; ++r) {
    for (unsigned c = 0; c < SrcCols; ++c) {
      if (c >= DstRows || r >= DstCols)
        continue;
      const unsigned src_idx = r * SrcCols + c;
      const unsigned dst_idx = c * DstCols + r;
      if (src_idx < kTileWords && dst_idx < kTileWords)
        out.words[dst_idx] = src.words[src_idx];
    }
  }
  (void)DType;
  (void)tmp;
  return out;
#else
  return __builtin_linx_tile_ttrans(src, tmp, SizeCode, DType, DstRows, DstCols,
                                   SrcRows, SrcCols);
#endif
}

} // namespace detail
} // namespace linx
} // namespace pto

#endif // PTO_LINX_IMPL_BACKEND_HPP
