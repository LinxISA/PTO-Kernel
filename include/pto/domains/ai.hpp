#ifndef PTO_DOMAINS_AI_HPP
#define PTO_DOMAINS_AI_HPP

namespace pto::domains::ai {

struct GemmTileProfile {
  int m;
  int n;
  int k;
};

inline constexpr GemmTileProfile default_gemm_profile() {
  return {16, 16, 4};
}

inline constexpr const char *name() { return "ai"; }

} // namespace pto::domains::ai

#endif // PTO_DOMAINS_AI_HPP
