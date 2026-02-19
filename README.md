# PTO-Kernel

PTO-Kernel is the LinxISA tile-kernel repository for high-performance AI, HPC,
rendering, and math workloads.

This repository hosts:
- PTO tile kernels (`kernels/`)
- PTO kernel tooling (`tools/`)
- Public PTO headers used by Linx tile kernels (`include/`)
- Domain API scaffolding (`include/pto/domains/`)
- Tile flow contracts and roadmap docs (`docs/`)
- Local development tests (`tests/`)

## Repository layout

- `include/`: public headers consumed by PTO kernels and integration tests
- `kernels/`: reference PTO kernels (GEMM, flash-attention, tile load/store, etc.)
- `tools/`: compile, parity, and asm-generation scripts
- `tests/`: host smoke tests and asm contract checks
- `docs/contracts/`: ISA/LLVM/QEMU alignment contracts for tile flow
- `docs/roadmap/`: staged LinxCore alignment plan
- `third_party/lib_pto/`: upstream licensing and notices for vendored PTO content

## Build (host smoke)

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Build (Linx cross asm checks)

```bash
cmake -S . -B build-linx -DPTO_ENABLE_LINX_CROSS=ON
cmake --build build-linx --target pto_linx_contracts
```

## Integration in LinxISA superproject

When mounted as `workloads/pto_kernels` submodule inside `linx-isa`, scripts
default to writing generated artifacts to `linx-isa/workloads/generated/` and
consume AVS/QEMU integration from the superproject.
