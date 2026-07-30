# Tile Op to ISA v0.57.1 Contract

This contract locks PTO tile-kernel intent to the PTO 0.57.1 operation catalog.
The normative identity source is the `PTO-ISA/pto-spec` revision recorded in
`pto_isa_v0571.lock.json`; generated projections are checked review artifacts.

## Required mapping surface

- The nine TMA operations lower to their exact typed commands:
  `BSTART.TLOAD`, `BSTART.TSTORE`, `BSTART.TMOV`, `BSTART.TPREFETCH`,
  `BSTART.MGATHER`, `BSTART.MSCATTER`, `BSTART.MGATHER.MASK`,
  `BSTART.MSCATTER.MASK`, and `BSTART.MGATHER.CAS`.
  Generic `BSTART.TMA` spelling is not an active compatibility path.
- The thirteen CUBE operations lower to distinct typed commands for
  `TMATMUL*`, `TMATMULMX*`, `TGEMV*`, `TGEMVMX*`, and `ACCCVT`. Retired
  textual `MAMULB` aliases and generic
  `BSTART.CUBE` spelling are forbidden in generated PTO kernel assembly.
- The 98 TEPL identities use the generated `Mode` and `Function` fields.
  Their raw selector is exactly `(Mode << 5) | Function`; no hand-maintained
  packed selector table is an authority.

## Legality anchors

- Canonical forms only (`V.*` + typed `BSTART.*` where applicable).
- `B.ARG` is retired. Typed TMA blocks use the applicable `B.IOR`, `B.DATR`,
  `B.DIM`, and `B.IOT` descriptors defined by the locked catalog.
- No legacy `L.*`, `MAMULB`, `BSTART.TMA`, or `BSTART.CUBE` aliases in
  canonical asm outputs.
- The direct API contains exactly 120 identities (98 TEPL, 9 TMA, 13 CUBE).
  Each identity has exactly the catalog operand order and arity; convenience
  overloads are not members of the direct `pto` namespace.
  `TADDC`, `TADDSC`, `TFMA`, `TFMOD`, `TFMODS`, `TLRELU`, `TRANDOM`,
  `TSUBC`, and `TSUBSC` are deleted, not compatibility aliases.
- `TTRANS` is canonical. `TTRANSPOSE` is not normalized by generators or
  coverage tools.
