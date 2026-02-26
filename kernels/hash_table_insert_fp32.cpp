#include <common/extended_kernel_runtime.hpp>

using namespace pto;

extern "C" void hash_table_insert_f32(int *table_keys_ptr,
                                       float *table_values_ptr, int table_size,
                                       int *keys_ptr, float *values_ptr,
                                       int n) {
  const int TS = table_size < 1 ? 1 : table_size;
  const int N = n < 1 ? 1 : n;

  for (int i = 0; i < N; ++i) {
    const int key = keys_ptr[i];
    const float val = values_ptr[i];

    int slot = -1;
    for (int t = 0; t < TS; ++t) {
      if (table_keys_ptr[t] == key) {
        slot = t;
        break;
      }
      if (slot < 0 && table_keys_ptr[t] == -1)
        slot = t;
    }
    if (slot >= 0) {
      table_keys_ptr[slot] = key;
      table_values_ptr[slot] = val;
    }
  }
}
