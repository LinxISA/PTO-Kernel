# Tile QEMU Parity Contract

This contract governs parity between host simulation and Linx QEMU execution.

## Required parity signal

- Each kernel emits a deterministic digest payload keyed by kernel name.
- Host and QEMU digests must match under smoke profile (`PTO_QEMU_SMOKE=1`).

## Integration assumptions

- AVS `pto_parity` suite is the integration runtime entry.
- Parity runner writes reports in superproject `workloads/generated/` when this
  repo is mounted as a submodule.
