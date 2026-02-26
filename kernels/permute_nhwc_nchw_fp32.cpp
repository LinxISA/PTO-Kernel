#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void permute_nhwc_nchw_f32(float *out_ptr, float *in_ptr, int n,
                                        int h, int w, int c) {
  const int N = n < 1 ? 1 : n;
  const int H = h < 1 ? 1 : h;
  const int W = w < 1 ? 1 : w;
  const int C = c < 1 ? 1 : c;

  for (int nn = 0; nn < N; ++nn) {
    for (int hh = 0; hh < H; ++hh) {
      for (int ww = 0; ww < W; ++ww) {
        for (int cc = 0; cc < C; ++cc) {
          const int src = ((nn * H + hh) * W + ww) * C + cc;
          const int dst = ((nn * C + cc) * H + hh) * W + ww;
          out_ptr[dst] = in_ptr[src];
        }
      }
    }
  }
}
