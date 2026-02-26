#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void decode_greedy_search_f32(int *token_out_ptr, float *logits_ptr,
                                           int batch, int vocab) {
  const int B = batch < 1 ? 1 : batch;
  const int V = vocab < 1 ? 1 : vocab;
  for (int b = 0; b < B; ++b) {
    int best = 0;
    float bestv = logits_ptr[b * V + 0];
    for (int v = 1; v < V; ++v) {
      const float x = logits_ptr[b * V + v];
      if (x > bestv) {
        bestv = x;
        best = v;
      }
    }
    token_out_ptr[b] = best;
  }
}
