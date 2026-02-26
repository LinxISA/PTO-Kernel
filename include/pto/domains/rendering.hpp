#ifndef PTO_DOMAINS_RENDERING_HPP
#define PTO_DOMAINS_RENDERING_HPP

namespace pto::domains::rendering {

struct ShadingTileProfile {
  int width;
  int height;
};

// Early rendering kernels typically use 32x32 element tiles to match 4KB tiles for
// common 32-bit element formats (float32, packed RGBA8-in-u32).
inline constexpr ShadingTileProfile default_shading_profile() {
  return {32, 32};
}

// Rendering-oriented tile/type conventions.
// Kept separate so we can evolve the conventions without changing the core PTO tile system.
#include <pto/domains/rendering_types.hpp>

inline constexpr const char *name() { return "rendering"; }

} // namespace pto::domains::rendering

#endif // PTO_DOMAINS_RENDERING_HPP
