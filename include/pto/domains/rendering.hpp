#ifndef PTO_DOMAINS_RENDERING_HPP
#define PTO_DOMAINS_RENDERING_HPP

namespace pto::domains::rendering {

struct ShadingTileProfile {
  int width;
  int height;
};

// Early rendering kernels use a single-column 1024-element tile (1024x1) to match
// the 4KB RawTile carrier for common 32-bit element formats.
inline constexpr ShadingTileProfile default_shading_profile() {
  return {1, 1024};
}

// Rendering-oriented tile/type conventions.
// Kept separate so we can evolve the conventions without changing the core PTO tile system.
#include <pto/domains/rendering_types.hpp>

inline constexpr const char *name() { return "rendering"; }

} // namespace pto::domains::rendering

#endif // PTO_DOMAINS_RENDERING_HPP
