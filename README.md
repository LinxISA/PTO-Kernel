# PTO Kernels

## Scope
`workloads/pto_kernels` is the LinxISA PTO tile-kernel workspace for kernel sources, tooling, and host-side validation flows.

## Upstream
- Repository: `https://github.com/LinxISA/PTO-Kernel`
- Merge-back target branch: `main`

## What This Submodule Owns
- PTO kernel sources (`kernels/`)
- Public PTO headers (`include/`)
- Kernel/tooling/test scaffolding (`tools/`, `tests/`, `docs/`)

## Canonical Build and Test Commands
Run from `/Users/zhoubot/linx-isa/workloads/pto_kernels`.

```bash
cmake -S . -B build
cmake --build build -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
ctest --test-dir build --output-on-failure

cmake -S . -B build-linx -DPTO_ENABLE_LINX_CROSS=ON
cmake --build build-linx --target pto_linx_contracts
```

## LinxISA Integration Touchpoints
- Submodule pinned by the superproject for PTO kernel bring-up
- Consumed by Linx tile flow contracts and integration tests
- Tooling and artifacts integrated with LinxISA AVS/runtime workflows

## Related Docs
- `/Users/zhoubot/linx-isa/docs/project/navigation.md`
- `/Users/zhoubot/linx-isa/docs/bringup/`
- `/Users/zhoubot/linx-isa/workloads/pto_kernels/docs/`

## Tidy-Up Status
- `done`: split kernel sources by operation family under `kernels/attention`, `kernels/matmul`, `kernels/normalization`, `kernels/elementwise`, `kernels/layout`, `kernels/indexing`, `kernels/routing`, `kernels/decode`, and `kernels/memory`
- `done`: replaced flat-path discovery with `kernels/catalog.txt` for CMake, benchmark manifest tooling, parity tooling, asm tooling, and QEMU suite wiring
- `done`: added shared runtime and environment setup headers in `include/common/runtime/kernel_env.hpp`, `include/common/runtime/kernel_shapes.hpp`, and `include/common/runtime/kernel_tiling.hpp`
- `done`: moved representative benchmarked tiled kernels onto parameterized shape and tiling policy instead of per-file hardcoded tile constants
- `done`: kept full-profile shapes large by default and preserved reduced smoke shapes for QEMU
- `done`: revalidated host tests, Linx asm contracts, and host-vs-QEMU parity after the refactor
- `in_progress`: migrate the remaining non-benchmarked kernels to the shared shape and tiling headers so every kernel family uses the same configuration surface
