#ifndef PTO_DOMAINS_MATH_HPP
#define PTO_DOMAINS_MATH_HPP

namespace pto::domains::math {

struct ReductionTileProfile {
  int lanes;
};

inline constexpr ReductionTileProfile default_reduction_profile() {
  return {32};
}

inline constexpr const char *name() { return "math"; }

} // namespace pto::domains::math

#endif // PTO_DOMAINS_MATH_HPP
