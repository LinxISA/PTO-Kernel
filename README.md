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
