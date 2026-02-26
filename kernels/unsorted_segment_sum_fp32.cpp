#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void unsorted_segment_sum_f32(float *out_ptr, int *segment_ids_ptr,
                                          float *data_ptr, int n,
                                          int num_segments) {
  const int N = n < 1 ? 1 : n;
  const int S = num_segments < 1 ? 1 : num_segments;

  for (int s = 0; s < S; ++s)
    out_ptr[s] = 0.0f;

  for (int i = 0; i < N; ++i) {
    const int sid = segment_ids_ptr[i];
    if (sid >= 0 && sid < S)
      out_ptr[sid] += data_ptr[i];
  }
}
