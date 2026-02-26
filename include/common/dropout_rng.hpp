#ifndef PTO_COMMON_DROPOUT_RNG_HPP
#define PTO_COMMON_DROPOUT_RNG_HPP

#include <stdint.h>

namespace pto {

struct DropoutRng {
  uint64_t state;

  explicit DropoutRng(uint64_t seed) : state(seed ? seed : 0x9e3779b97f4a7c15ull) {}

  uint32_t next_u32() {
    uint64_t x = state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state = x;
    return static_cast<uint32_t>((x * 0x2545F4914F6CDD1Dull) >> 32);
  }

  float next_unit() {
    const uint32_t r = next_u32() >> 8; // 24-bit fraction
    return static_cast<float>(r) * (1.0f / 16777216.0f);
  }
};

inline bool dropout_keep(DropoutRng &rng, float keep_prob) {
  if (keep_prob <= 0.0f)
    return false;
  if (keep_prob >= 1.0f)
    return true;
  return rng.next_unit() < keep_prob;
}

} // namespace pto

#endif // PTO_COMMON_DROPOUT_RNG_HPP
