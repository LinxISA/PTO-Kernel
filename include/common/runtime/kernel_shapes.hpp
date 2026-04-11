#ifndef PTO_COMMON_RUNTIME_KERNEL_SHAPES_HPP
#define PTO_COMMON_RUNTIME_KERNEL_SHAPES_HPP

#include <common/runtime/kernel_env.hpp>

namespace pto {
namespace kernels {
namespace shapes {

inline constexpr int kMemoryRows = env::select(32, 1024);
inline constexpr int kMemoryCols = env::select(32, 1024);

inline constexpr int kMatmulM = env::select(16, 256);
inline constexpr int kMatmulN = env::select(16, 256);
inline constexpr int kMatmulK = env::select(16, 256);
inline constexpr int kMatmulReuseExtent = env::select(16, 64);

inline constexpr int kAttentionSeq = env::select(16, 128);
inline constexpr int kAttentionLargeSeq = env::select(16, 256);
inline constexpr int kAttentionQD = 16;
inline constexpr int kAttentionVD = 16;
inline constexpr int kAttentionSmallQD = 4;

inline constexpr int kMlaInputDim = 16;
inline constexpr int kMlaLatentDim = 4;
inline constexpr int kMlaOutputDim = 16;

inline constexpr int kNormTokens = env::select(16, 128);

inline constexpr int kSmallVector = env::select(64, 1024);
inline constexpr int kSmallTable = env::select(97, 2048);

} // namespace shapes
} // namespace kernels
} // namespace pto

#endif // PTO_COMMON_RUNTIME_KERNEL_SHAPES_HPP
