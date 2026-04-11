#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void decode_temperature_scaling_f32(float *scaled_ptr,
                                                 float *logits_ptr, int n,
                                                 float temperature) {
  const int N = n < 1 ? 1 : n;
  const float t = (temperature <= 1e-6f) ? 1e-6f : temperature;
  for (int i = 0; i < N; ++i)
    scaled_ptr[i] = logits_ptr[i] / t;
}
