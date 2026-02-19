# Tile Op to ISA v0.3 Contract

This contract locks PTO tile-kernel intent to strict LinxISA v0.3 block forms.

## Required mapping surface

- `TLOAD/TSTORE` must lower to `BSTART.TLOAD` / `BSTART.TSTORE` (or typed
  `BSTART.TMA` function aliases) with valid descriptor sequence.
- Matrix ops (`TMATMUL`, `TMATMUL_ACC`) must lower to typed cube/tma forms
  accepted by strict v0.3 legality rules.
- Template vector ops must remain in TEPL expansion space (`BSTART.TEPL`
  families), not legacy aliases.

## Legality anchors

- Canonical forms only (`V.*` + typed `BSTART.*` where applicable).
- Tile metadata descriptors (`B.ARG`, `B.IOR`, `B.IOT`) required for TMA
  data movement blocks.
- No legacy `L.*` aliases in canonical asm outputs.
