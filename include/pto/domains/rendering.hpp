#ifndef PTO_DOMAINS_RENDERING_HPP
#define PTO_DOMAINS_RENDERING_HPP

namespace pto::domains::rendering {

struct ShadingTileProfile {
  int width;
  int height;
};

inline constexpr ShadingTileProfile default_shading_profile() {
  return {16, 16};
}

inline constexpr const char *name() { return "rendering"; }

} // namespace pto::domains::rendering

#endif // PTO_DOMAINS_RENDERING_HPP
