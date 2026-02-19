#ifndef PTO_DOMAINS_HPC_HPP
#define PTO_DOMAINS_HPC_HPP

namespace pto::domains::hpc {

struct StencilTileProfile {
  int rows;
  int cols;
};

inline constexpr StencilTileProfile default_stencil_profile() {
  return {32, 32};
}

inline constexpr const char *name() { return "hpc"; }

} // namespace pto::domains::hpc

#endif // PTO_DOMAINS_HPC_HPP
