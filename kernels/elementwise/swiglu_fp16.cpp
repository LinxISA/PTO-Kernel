#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void swiglu_f16(fp16_t *out_ptr, fp16_t *x_ptr, fp16_t *gate_ptr,
                            int n) {
  const int N = n < 1 ? 1 : n;
  for (int i = 0; i < N; ++i) {
    const float x = fp16_to_float(x_ptr[i]);
    const float g = fp16_to_float(gate_ptr[i]);
    const float sg = g / (1.0f + kernels::m_exp(-kernels::clampf(g, -20.0f, 20.0f)));
    out_ptr[i] = float_to_fp16(x * sg);
  }
}
