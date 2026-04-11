# PTO Kernels for LinxISA

This folder contains PTO tile kernels compiled through the LinxISA LLVM backend.

Naming policy:
- kernel source names do not use the legacy `pto_` prefix.
- kernel source names do not use the legacy `_auto` suffix.

Layout:
- `attention/`: attention, FlashMLA, paged attention, sparse attention, RoPE, dropout
- `matmul/`: matmul, GEMM, reuse variants, quantized matmul
- `normalization/`: RMSNorm, LayerNorm, BatchNorm
- `elementwise/`: add, GELU, ReLU, Sigmoid, SiLU, Softmax, SwigLU, Tanh
- `layout/`: concat, split, stack, transpose, permute, reshape-family
- `indexing/`: argmax, gather, scatter, segment, unique, hash-table helpers
- `routing/`: MoE routing and MLP primitives
- `decode/`: decode filters and search helpers
- `memory/`: raw tile load/store coverage kernels

Catalog:
- `catalog.txt` is the source of truth for kernel discovery.
- build scripts, manifest tooling, parity tooling, and QEMU suite wiring all resolve sources through the catalog instead of assuming a flat directory.

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

Shared env and tiling policy:
- `include/common/runtime/kernel_env.hpp` owns `PTO_QEMU_SMOKE` and `PTO_USE_MIXED_TILE_SIMT`.
- `include/common/runtime/kernel_shapes.hpp` holds the default large-shape and smoke-shape presets.
- `include/common/runtime/kernel_tiling.hpp` holds parameterized tile sizes such as `PTO_GEMM_TILE_*`, `PTO_FLASH_TILE_*`, `PTO_FLASH_VEC_*`, and `PTO_FLASH_CUBE_*`.
- kernels are grouped by operation kind, not by dtype.

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
