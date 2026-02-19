# LinxCore Alignment Roadmap

This roadmap tracks PTO tile-flow convergence from kernel source to eventual
LinxCore execution fidelity.

## Near-term

- Keep strict v0.3 legal block emission validated through LLVM + asm checks.
- Maintain host-vs-QEMU parity as the required integration signal.

## Mid-term

- Add digest comparers between QEMU traces and LinxCore trace exports.
- Add boundary-level checks for typed tile block commit semantics.

## Long-term

- Enforce fail-fast checks for non-canonical block headers in full stack CI.
- Stabilize throughput-oriented kernel suites for AI/HPC/rendering/math domains.
