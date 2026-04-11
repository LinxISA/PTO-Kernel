#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void matmul_a8w8(int32_t *out_ptr, int8_t *a_ptr, int8_t *w_ptr,
                             int m, int n, int k) {
  const int M = m < 1 ? 1 : m;
  const int N = n < 1 ? 1 : n;
  const int K = k < 1 ? 1 : k;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t acc = 0;
      for (int kk = 0; kk < K; ++kk)
        acc += static_cast<int32_t>(a_ptr[i * K + kk]) *
               static_cast<int32_t>(w_ptr[j * K + kk]);
      out_ptr[i * N + j] = acc;
    }
  }
}
