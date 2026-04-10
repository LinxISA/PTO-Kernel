#ifndef PTO_COMMON_BLOCK_VECTOR_COMPAT_HPP
#define PTO_COMMON_BLOCK_VECTOR_COMPAT_HPP

#include <common/pto_tileop.hpp>

#include <stddef.h>
#include <stdint.h>
#ifndef __vec__
#define __vec__ inline __attribute__((always_inline))
#endif

#ifndef __vpar__
#define __vpar__ inline __attribute__((always_inline)) \
  __attribute__((annotate("linx_vpar_kernel")))
#endif

#ifndef __mpar__
#define __mpar__ inline __attribute__((always_inline)) \
  __attribute__((annotate("linx_mpar_kernel")))
#endif

#ifndef __out__
#define __out__
#endif

#ifndef __in__
#define __in__
#endif

#ifndef __vbuf__
#if defined(__ubuf__)
#define __vbuf__ __ubuf__
#else
#define __vbuf__
#endif
#endif

namespace pto {
namespace blkv {
namespace detail {

struct LaunchState {
  size_t x = 0;
  size_t y = 0;
  size_t z = 0;
};

// The mixed block-vector compatibility path is single-threaded in both host
// simulation and the current freestanding Linx/QEMU lane. Avoid TLS here so
// the fallback kernels do not depend on bare-metal thread-local runtime setup.
inline LaunchState g_launch_state{};

inline float exp_approx(float x) {
  if (x < -10.0f)
    return 0.0f;
  if (x > 10.0f)
    x = 10.0f;
  float term = 1.0f;
  float sum = 1.0f;
  for (int i = 1; i <= 7; ++i) {
    term *= x / static_cast<float>(i);
    sum += term;
  }
  return sum > 0.0f ? sum : 0.0f;
}

inline float sqrt_approx(float x) {
  if (x <= 0.0f)
    return 0.0f;
  float guess = x > 1.0f ? x : 1.0f;
  for (int i = 0; i < 6; ++i)
    guess = 0.5f * (guess + x / guess);
  return guess;
}

template <typename TileT>
inline typename TileT::DType *tile_ptr(TileT &tile) {
  return reinterpret_cast<typename TileT::DType *>(&tile.raw());
}

template <typename TileT>
inline const typename TileT::DType *tile_ptr(const TileT &tile) {
  return reinterpret_cast<const typename TileT::DType *>(&tile.raw());
}

} // namespace detail

#if defined(__LINXISA__) && !defined(PTO_HOST_SIM)
inline uint16_t blkv_get_index_x() {
  return static_cast<uint16_t>(__builtin_linx_lc0());
}
inline uint16_t blkv_get_index_y() {
  return static_cast<uint16_t>(__builtin_linx_lc1());
}
inline uint16_t blkv_get_index_z() {
  return static_cast<uint16_t>(__builtin_linx_lc2());
}
#else
inline uint16_t blkv_get_index_x() {
  return static_cast<uint16_t>(detail::g_launch_state.x);
}
inline uint16_t blkv_get_index_y() {
  return static_cast<uint16_t>(detail::g_launch_state.y);
}
inline uint16_t blkv_get_index_z() {
  return static_cast<uint16_t>(detail::g_launch_state.z);
}
#endif

template <typename TileT>
inline typename TileT::DType *blkv_get_tile_ptr(TileT *tile) {
  return detail::tile_ptr(*tile);
}

template <typename TileT>
inline const typename TileT::DType *blkv_get_tile_ptr(const TileT *tile) {
  return detail::tile_ptr(*tile);
}

template <typename TileT>
inline typename TileT::DType *blkv_get_tile_ptr(TileT &tile) {
  return detail::tile_ptr(tile);
}

template <typename TileT>
inline const typename TileT::DType *blkv_get_tile_ptr(const TileT &tile) {
  return detail::tile_ptr(tile);
}

template <typename T>
inline T blkv_max(T a, T b) {
  return a < b ? b : a;
}

inline float blkv_fexp(float x) { return detail::exp_approx(x); }
inline float blkv_fsqrt(float x) { return detail::sqrt_approx(x); }

template <typename Fn>
inline void blkv_for_1d(size_t dim_x, Fn &&fn) {
  for (size_t x = 0; x < dim_x; ++x) {
    detail::g_launch_state.x = x;
    detail::g_launch_state.y = 0;
    detail::g_launch_state.z = 0;
    fn();
  }
}

template <typename Fn>
inline void blkv_for_2d(size_t dim_x, size_t dim_y, Fn &&fn) {
  for (size_t x = 0; x < dim_x; ++x) {
    for (size_t y = 0; y < dim_y; ++y) {
      detail::g_launch_state.x = x;
      detail::g_launch_state.y = y;
      detail::g_launch_state.z = 0;
      fn();
    }
  }
}

} // namespace blkv
} // namespace pto

#endif // PTO_COMMON_BLOCK_VECTOR_COMPAT_HPP
