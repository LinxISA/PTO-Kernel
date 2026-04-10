#include <common/block_vector_kernels.hpp>
#include <common/extended_kernel_runtime.hpp>

int main() {
  using TileW = pto::Tile<pto::Location::Vec, float, 2, 4, pto::BLayout::RowMajor>;
  using TileMax = pto::Tile<pto::Location::Vec, float, 2, 1, pto::BLayout::RowMajor>;
  using TileSum = pto::Tile<pto::Location::Vec, float, 2, 1, pto::BLayout::RowMajor>;

  TileW src0;
  TileW src1;
  TileW exp0;
  TileW exp1;
  TileMax old_max;
  TileMax new_max;
  TileMax scale;
  TileSum old_sum;
  TileSum new_sum;

  float *src0_ptr = pto::blkv::blkv_get_tile_ptr(src0.data());
  float *src1_ptr = pto::blkv::blkv_get_tile_ptr(src1.data());
  float *old_max_ptr = pto::blkv::blkv_get_tile_ptr(old_max.data());
  float *old_sum_ptr = pto::blkv::blkv_get_tile_ptr(old_sum.data());

  for (int i = 0; i < 8; ++i) {
    src0_ptr[i] = static_cast<float>(i + 1);
    src1_ptr[i] = static_cast<float>(8 - i);
  }
  old_max_ptr[0] = -1e30f;
  old_max_ptr[1] = -1e30f;
  old_sum_ptr[0] = 0.0f;
  old_sum_ptr[1] = 0.0f;

  pto::blkv::blkv_for_1d(TileMax::ValidRow, [&] {
    pto::kernels::new_max_2src<TileW, TileMax>(
        scale.data(), new_max.data(), src0.data(), src1.data(), old_max.data(),
        1.0f);
  });
  pto::blkv::blkv_for_1d(TileSum::ValidRow, [&] {
    pto::kernels::src_exp_2src_with_new_sum<TileW, TileW, TileMax, TileSum, TileMax>(
        new_sum.data(), exp0.data(), exp1.data(), src0.data(), src1.data(),
        new_max.data(), old_sum.data(), scale.data(), 1.0f);
  });

  const float *new_max_ptr = pto::blkv::blkv_get_tile_ptr(new_max.data());
  const float *new_sum_ptr = pto::blkv::blkv_get_tile_ptr(new_sum.data());
  if (new_max_ptr[0] != 8.0f || new_max_ptr[1] != 8.0f)
    return 1;
  if (!(new_sum_ptr[0] > 0.0f) || !(new_sum_ptr[1] > 0.0f))
    return 2;

  TileW src2;
  TileW src3;
  TileW exp2;
  TileW exp3;
  TileSum local_sum_0;
  TileSum local_sum_1;

  float *src2_ptr = pto::blkv::blkv_get_tile_ptr(src2.data());
  float *src3_ptr = pto::blkv::blkv_get_tile_ptr(src3.data());
  for (int i = 0; i < 8; ++i) {
    src2_ptr[i] = static_cast<float>(i + 2);
    src3_ptr[i] = static_cast<float>(10 - i);
  }

  pto::blkv::blkv_for_1d(TileMax::ValidRow, [&] {
    pto::kernels::new_max_4src<TileW, TileMax>(
        scale.data(), new_max.data(), src0.data(), src1.data(), src2.data(),
        src3.data(), old_max.data(), 1.0f);
  });
  pto::blkv::blkv_for_1d(TileSum::ValidRow, [&] {
    pto::kernels::src_exp_2src_with_local_sum<TileW, TileW, TileMax, TileSum>(
        local_sum_0.data(), exp0.data(), exp1.data(), src0.data(), src1.data(),
        new_max.data(), 1.0f);
    pto::kernels::src_exp_2src_with_local_sum<TileW, TileW, TileMax, TileSum>(
        local_sum_1.data(), exp2.data(), exp3.data(), src2.data(), src3.data(),
        new_max.data(), 1.0f);
    pto::kernels::new_sum_of_2_loc_sum<TileMax, TileSum>(
        new_sum.data(), local_sum_0.data(), local_sum_1.data(), old_sum.data(),
        scale.data());
  });

  if (new_max_ptr[0] != 10.0f || new_max_ptr[1] != 9.0f)
    return 3;
  if (!(new_sum_ptr[0] > 0.0f) || !(new_sum_ptr[1] > 0.0f))
    return 4;

  alignas(64) float q[16 * 16];
  alignas(64) float k[16 * 16];
  alignas(64) float v[16 * 16];
  alignas(64) float mixed_out[16 * 16];
  alignas(64) float ref_out[16 * 16];

  for (int i = 0; i < 16 * 16; ++i) {
    q[i] = static_cast<float>((i % 11) - 5) * 0.125f;
    k[i] = static_cast<float>((i % 7) - 3) * 0.25f;
    v[i] = static_cast<float>((i % 13) - 6) * 0.0625f;
  }

  pto::kernels::mixed_attention_f32<16, 16, 16, 8, 4, 4, false>(
      mixed_out, q, k, v);
  pto::kernels::dense_attention_f32<16>(q, k, v, ref_out, 16, 16, 16, false);

  for (int i = 0; i < 16 * 16; ++i) {
    const float diff = pto::kernels::m_abs(mixed_out[i] - ref_out[i]);
    if (diff > 6.0e-2f)
      return 5;
  }

  return 0;
}
