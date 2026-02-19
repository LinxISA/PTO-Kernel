# Tile LLVM Codegen Contract

This contract defines expectations for Linx LLVM lowering from PTO kernels.

## Compiler expectations

- PTO kernels compile under Linx target triples with deterministic tile header
  emission.
- Tile block descriptors are emitted in legal order for strict v0.3.
- Compile-time options can disable unstable SIMT autovec paths without changing
  PTO kernel source semantics.

## Contract checks

- Cross-compiled asm must include required `BSTART.T*` forms.
- Forbidden legacy mnemonics must not appear in generated asm.
- Tile register-group usage and block descriptor coverage checks are enforced by
  tooling scripts.
