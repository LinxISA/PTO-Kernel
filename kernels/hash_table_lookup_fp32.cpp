#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void hash_table_lookup_f32(float *out_values_ptr, int *keys_ptr,
                                       int *table_keys_ptr,
                                       float *table_values_ptr, int table_size,
                                       int nkeys) {
  const int TS = table_size < 1 ? 1 : table_size;
  const int NK = nkeys < 1 ? 1 : nkeys;

  for (int i = 0; i < NK; ++i) {
    const int key = keys_ptr[i];
    float v = 0.0f;
    bool found = false;
    for (int t = 0; t < TS; ++t) {
      if (table_keys_ptr[t] == key) {
        v = table_values_ptr[t];
        found = true;
        break;
      }
    }
    out_values_ptr[i] = found ? v : 0.0f;
  }
}
