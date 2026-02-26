# PTO Kernels for LinxISA (v0.3)

This folder contains PTO tile kernels compiled through the LinxISA LLVM backend.

Naming policy:
- kernel source names do not use the legacy `pto_` prefix.
- kernel source names do not use the legacy `_auto` suffix.

Kernels:
- `tload_store.cpp`
- `mamulb.cpp`
- `tmatmul_acc.cpp`
- `gemm.cpp`
- `gemm_basic.cpp`
- `gemm_demo.cpp`
- `gemm_performance.cpp`
- `add_custom.cpp`
- `flash_attention.cpp`
- `flash_attention_demo.cpp`
- `flash_attention_masked.cpp`
- `fa_performance.cpp`
- `mla_attention_demo.cpp`
- `flash_attention_cube_fp16.cpp`
- `flash_attention_cube_fp8_e4m3.cpp`
- `flash_attention_cube_fp4_e2m1.cpp`
- `flash_attention_vec_fp32.cpp`
- `flash_attention_vec_fp16.cpp`
- `mha_fp16.cpp`
- `gqa_fp16.cpp`
- `rope_apply_fp16.cpp`
- `attention_dropout_fp16.cpp`
- `flash_mla_deepseekv3_fp16.cpp`
- `flash_mla_deepseekv3_fp8_e4m3.cpp`
- `ifa_mla_seq1_fp16.cpp`
- `ifa_gqa_seq1_fp16.cpp`
- `paged_attention_mha_fp16.cpp`
- `paged_attention_gqa_fp16.cpp`
- `moe_sort_fp32.cpp`
- `moe_topk_fp32.cpp`
- `moe_gate_route_fp16.cpp`
- `moe_mlp_fp16.cpp`
- `gemm_reuse_a_fp16.cpp`
- `gemm_reuse_b_fp16.cpp`
- `gemm_reuse_ab_fp16.cpp`
- `flash_attention_backward_fp16.cpp`
- `flash_attention_backward_fp32.cpp`
- `sparse_attention_local_fp16.cpp`
- `sparse_attention_block_fp16.cpp`
- `rmsnorm_fp16.cpp`
- `relu_fp32.cpp`
- `sigmoid_fp32.cpp`
- `softmax_fp32.cpp`
- `tanh_fp32.cpp`
- `silu_fp32.cpp`
- `gelu_fp32.cpp`
- `swiglu_fp16.cpp`
- `layernorm_fp16.cpp`
- `batchnorm_fp16.cpp`
- `argmax_fp32.cpp`
- `matmul_a8w8.cpp`
- `matmul_a8w4.cpp`
- `matmul_a16w8.cpp`
- `matmul_a16w4.cpp`
- `decode_greedy_search_fp32.cpp`
- `decode_beam_search_fp32.cpp`
- `decode_temperature_scaling_fp32.cpp`
- `decode_topk_filter_fp32.cpp`
- `decode_topp_nucleus_filter_fp32.cpp`
- `transpose_large_fp32.cpp`
- `permute_nhwc_nchw_fp32.cpp`
- `slice_fp32.cpp`
- `gather_fp32.cpp`
- `scatter_fp32.cpp`
- `where_fp32.cpp`
- `concat_fp32.cpp`
- `split_fp32.cpp`
- `stack_fp32.cpp`
- `reshape_fp32.cpp`
- `flatten_fp32.cpp`
- `squeeze_fp32.cpp`
- `unsqueeze_fp32.cpp`
- `hash_table_lookup_fp32.cpp`
- `hash_table_insert_fp32.cpp`
- `unsorted_segment_sum_fp32.cpp`
- `unique_i32.cpp`

All kernels:
- include `common/pto_tileop.hpp` from `workloads/pto_kernels/include` (when
  mounted as LinxISA submodule),
- low-precision kernels also use `common/linx_lowp_types.hpp` for deterministic
  fp16/fp8/fp4 conversion policy and `common/dropout_rng.hpp` for dropout masks,
- use `global_tensor` + `global_iterator` addressing for TLOAD/TSTORE shape/stride inference,
- iterate over large tensors with nested tile loops,
- obey strict tile-byte legality (`<=4KB`) for both tile descriptors and TMATMUL footprints using
  `tile_bytes = ceil(dim0*dim1*dim2*elem_bits/8)` (`dim2=1` when absent).
- use strict-v0.3 DataType encoding (`FP64/FP32/FP16/FP8/BF16/FPL8/FP4/FPL4`,
  `INT64/INT32/INT16/INT8/INT4`, `UINT64/UINT32/UINT16/UINT8/UINT4`) in compiler and runtime checks.

Runtime profile policy:
- default full profile keeps original larger tensor domains for compile/asm bring-up.
- `PTO_QEMU_SMOKE=1` enables reduced runtime domains for QEMU execution while preserving tile-op and loop-path coverage.
- masked kernels keep non-zero remainder paths in smoke profile.

Parity gate:
- host-vs-QEMU parity runner:
  `/Users/zhoubot/linx-isa/workloads/pto_kernels/tools/run_pto_kernel_parity.py`
- report artifacts:
  - `/Users/zhoubot/linx-isa/workloads/generated/pto_kernel_parity_latest.json`
  - `/Users/zhoubot/linx-isa/workloads/generated/pto_kernel_parity_latest.md`

Objdump artifacts:
- per-kernel objects and disassembly are generated under:
  - `/Users/zhoubot/linx-isa/workloads/generated/pto_objdump/obj`
  - `/Users/zhoubot/linx-isa/workloads/generated/pto_objdump/dis`
