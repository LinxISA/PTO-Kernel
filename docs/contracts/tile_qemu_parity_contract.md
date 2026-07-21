# Tile QEMU Parity Contract

This contract governs parity between host simulation and Linx QEMU execution.

## Required parity signals

- Every API in `docs/upstream/deepseek-tilekernels.json` must be declared,
  implemented, invoked by the QEMU AVS, and paired with a numerical or
  structural `PTO-ORACLE` contract in the AVS source.
- Every PTO operation reachable from those kernels must exist in the frozen
  v0.57 catalog, have a QEMU semantic selector case, and appear in the Linx
  assembly generated from the kernel sources.
- Quantization paths require scalar reconstruction checks; positive scales or
  nonzero digests alone are not correctness evidence.
- Host simulation and QEMU must both pass after a kernel or backend change.

`tests/check_deepseek_verification_coverage.py` is the machine gate for these
denominators. Its JSON report intentionally distinguishes the full 111-op ISA
catalog from the kernel-reachable subset. A 32/32 kernel result must never be
reported as 111/111 executable ISA semantic coverage.

## Integration assumptions

- AVS `pto_parity` suite is the integration runtime entry.
- AVS `deepseek_tilekernels` is the numerical runtime-oracle entry for the 43
  upstream DeepSeek APIs.
- CTest `pto_deepseek_kernel_host_oracle` is the host-simulation reference
  entry for library/backend gaps found by those runtime oracles.
- CMake target `pto_deepseek_linx_contracts` emits
  `deepseek-verification-coverage.json` after assembly and selector checks.
- Parity runner writes reports in superproject `workloads/generated/` when this
  repo is mounted as a submodule.
