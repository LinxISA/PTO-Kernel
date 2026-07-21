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

# Add -DPTO_BENCHMARK_WORKBOOK=/path/to/benchmark_master_list_completed.xlsx
# to enable the workbook-backed manifest test.

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
- Pages dashboard: `https://linxisa.github.io/pto-kernel/` (published from `docs/`)
- DeepSeek TileKernels provenance and source-to-PTO mapping:
  `docs/upstream/deepseek-tilekernels.json`

## DeepSeek TileKernels lane

The `kernels/upstream/deepseek/` lane ports all 37 kernel source files from
`deepseek-ai/TileKernels` main at the exact commit recorded in the provenance
manifest. The port is freestanding and has no Python, PyTorch, CUDA, or
TileLang runtime dependency. Kernel data paths are loop-free C++ compositions
of named PTO ISA v0.57 intrinsics. Scalar iteration is limited to tile-grid
control in the shared helper; element data paths remain PTO operations. The
physical carrier is 32x32 row-major, while runtime dimensions drive rectangular
tail masks (`LB0=ValidCol`, `LB1=ValidRow`, `LB2=physical Col`) and row stride.
The QEMU suite covers both the 32x32 standard profile and non-multiple tail
shapes, including a multi-tile 35x37 transpose.

Run the host contracts locally:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

From the LinxISA superproject, compile and execute the standard-input suite:

```bash
python3 avs/qemu/run_tests.py --suite deepseek_tilekernels --verbose
```

## Tidy-Up Status
- `done`: split kernel sources by operation family under `kernels/attention`, `kernels/matmul`, `kernels/normalization`, `kernels/elementwise`, `kernels/layout`, `kernels/indexing`, `kernels/routing`, `kernels/decode`, and `kernels/memory`
- `done`: replaced flat-path discovery with `kernels/catalog.txt` for CMake, benchmark manifest tooling, parity tooling, asm tooling, and QEMU suite wiring
- `done`: added shared runtime and environment setup headers in `include/common/runtime/kernel_env.hpp`, `include/common/runtime/kernel_shapes.hpp`, and `include/common/runtime/kernel_tiling.hpp`
- `done`: moved representative benchmarked tiled kernels onto parameterized shape and tiling policy instead of per-file hardcoded tile constants
- `done`: kept full-profile shapes large by default and preserved reduced smoke shapes for QEMU
- `done`: revalidated host tests, Linx asm contracts, and host-vs-QEMU parity after the refactor
- `in_progress`: migrate the remaining non-benchmarked kernels to the shared shape and tiling headers so every kernel family uses the same configuration surface

## Status Panel
- site source: `docs/index.html`, `docs/assets/site.css`, `docs/assets/site.js`
- generated data: `docs/data/status.json`
- refresh command: `python3 tools/build_status_site.py --workbook /path/to/benchmark_master_list_completed.xlsx`
- manifest command: `python3 tools/benchmark_manifest.py --workbook /path/to/benchmark_master_list_completed.xlsx`
- the workbook is always an explicit input; repository tools never search a home directory for it
