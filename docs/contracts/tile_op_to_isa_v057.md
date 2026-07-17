# Tile Op to ISA v0.57 Contract

This contract locks PTO tile-kernel intent to strict LinxISA v0.57 block forms.

## Required mapping surface

- `TLOAD/TSTORE/TMOV/TPREFETCH` must lower to exact named TMA aliases such as
  `BSTART.TLOAD`, `BSTART.TSTORE`, `BSTART.TMOV`, and `BSTART.TPREFETCH`.
  Generic `BSTART.TMA` spelling is not an active compatibility path.
- Matrix ops (`TMATMUL`, `TMATMUL_ACC`, and MX/bias variants) must lower to
  exact named cube aliases such as `BSTART.TMATMUL`, `BSTART.TMATMUL.ACC`,
  `BSTART.TMATMUL.BIAS`, `BSTART.TMATMULMX`, `BSTART.TMATMULMX.ACC`, and
  `BSTART.TMATMULMX.BIAS`. Retired textual `MAMULB` aliases and generic
  `BSTART.CUBE` spelling are forbidden in generated PTO kernel assembly.
- Template vector ops must remain in TEPL expansion space (`BSTART.TEPL`
  families), not legacy aliases.

## Legality anchors

- Canonical forms only (`V.*` + typed `BSTART.*` where applicable).
- Tile metadata descriptors (`B.ARG`, `B.IOR`, `B.IOT`) required for TMA
  data movement blocks.
- No legacy `L.*`, `MAMULB`, `BSTART.TMA`, or `BSTART.CUBE` aliases in
  canonical asm outputs.
