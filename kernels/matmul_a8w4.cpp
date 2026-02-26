#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void matmul_a8w4(int32_t *out_ptr, int8_t *a_ptr, fp4_e2m1_t *w_ptr,
                             int m, int n, int k) {
  const int M = m < 1 ? 1 : m;
  const int N = n < 1 ? 1 : n;
  const int K = k < 1 ? 1 : k;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        const float a = static_cast<float>(a_ptr[i * K + kk]);
        const float w = fp4_e2m1_to_float(w_ptr[j * K + kk]);
        acc += a * w;
      }
      out_ptr[i * N + j] = static_cast<int32_t>(acc);
    }
  }
}
