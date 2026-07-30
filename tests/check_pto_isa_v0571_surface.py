#!/usr/bin/env python3
"""Fail-closed checks for the exact PTO 0.57.1 direct operation surface."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SURFACE = ROOT / "docs/contracts/generated/pto_isa_v0571_surface.json"
BASE_API = ROOT / "include/pto/common/pto_instr.hpp"
GENERATED_API = ROOT / "include/pto/common/generated/pto_isa_v0571_api.inc"
TILEOP = ROOT / "include/common/pto_tileop.hpp"
BACKEND = ROOT / "include/pto/linx/impl/backend.hpp"
PTO_IMPL = ROOT / "include/pto/linx/impl/PtoInstrImpl.hpp"
LOW_LEVEL_TILEOPS = ROOT / "include/pto/linx/TileOps.hpp"

DELETED = {
    "TADDC", "TADDSC", "TFMA", "TFMOD", "TFMODS", "TLRELU",
    "TRANDOM", "TSUBC", "TSUBSC",
}
API_MARKER_RE = re.compile(r"^// PTO-ISA-API: ([A-Z][A-Z0-9_]*)\(([^)]*)\)$", re.MULTILINE)
PTO_INST_RE = re.compile(r"PTO_INST\b(?:(?!\{).)*?\b([A-Z][A-Z0-9_]*)\s*\(", re.DOTALL)


def main() -> int:
    errors: list[str] = []
    document = json.loads(SURFACE.read_text(encoding="utf-8"))
    operations = document.get("operations", [])
    names = [str(item["name"]) for item in operations]
    expected_api = [
        (str(item["name"]), tuple(str(op["field"]) for op in item.get("operands", [])))
        for item in operations
    ]
    families = Counter(str(item["family"]) for item in operations)

    if len(names) != 120 or len(set(names)) != 120:
        errors.append("direct surface must contain exactly 120 unique operations")
    if families != {"TEPL": 98, "TMA": 9, "CUBE": 13}:
        errors.append(f"family counts differ from 98/9/13: {dict(families)}")
    if set(document.get("deleted_names", [])) != DELETED:
        errors.append("deleted-name set differs from the PTO 0.57.1 contract")

    for item in operations:
        if item["family"] == "TEPL":
            selector = int(str(item["selector"]), 16)
            packed = (int(item["mode"]) << 5) | int(item["function"])
            if selector != packed:
                errors.append(f"{item['name']}: selector {selector:#05x} != {packed:#05x}")

    generated = GENERATED_API.read_text(encoding="utf-8")
    actual_api = []
    for name, operands in API_MARKER_RE.findall(generated):
        actual_api.append((name, tuple(x.strip() for x in operands.split(",") if x.strip())))
    if actual_api != expected_api:
        errors.append("generated API name/arity/operand order differs from pto-spec catalog")
    generated_names = PTO_INST_RE.findall(generated)
    if generated_names != names:
        errors.append("generated API must define every canonical operation exactly once in catalog order")
    for name in names:
        if generated.count(f"dispatch<{name}Operation>") != 1:
            errors.append(f"{name}: direct API does not dispatch through its exact identity tag")

    base = BASE_API.read_text(encoding="utf-8")
    if re.search(r"namespace\s+legacy\s*\{", base):
        errors.append("legacy convenience API namespace remains active")
    leaked = set(PTO_INST_RE.findall(base)) & set(names)
    if leaked:
        errors.append(f"hand-written canonical overloads remain in namespace pto: {sorted(leaked)}")

    typed = [item for item in operations if item["family"] in {"TMA", "CUBE"}]
    commands = [str(item["command_mnemonic"]) for item in typed]
    if len(commands) != 22 or len(set(commands)) != 22:
        errors.append("TMA/CUBE must expose 22 unique typed command identities")
    if {"BSTART.TMA", "BSTART.CUBE"} & set(commands):
        errors.append("generic TMA/CUBE command leaked into direct surface")
    cube_functions = {int(item["function"]) for item in operations if item["family"] == "CUBE"}
    if cube_functions != {0, 1, 2, 4, 5, 6, 8, 16, 17, 18, 20, 21, 22}:
        errors.append(f"CUBE function identities differ: {sorted(cube_functions)}")

    tileop = TILEOP.read_text(encoding="utf-8")
    backend = BACKEND.read_text(encoding="utf-8")
    impl = PTO_IMPL.read_text(encoding="utf-8")
    low_level = LOW_LEVEL_TILEOPS.read_text(encoding="utf-8")
    if "common/generated/pto_isa_v0571.hpp" not in tileop:
        errors.append("Linx tile wrapper does not include generated selectors")
    if re.search(r"tepl(?:Unary|Binary|Splat)<0x[0-9A-Fa-f]+", tileop):
        errors.append("Linx tile wrapper contains a raw TEPL selector")
    if "TRECIP_IMPL" not in impl or re.search(r"TRECIP_IMPL[\s\S]{0,180}TDIVS\s*\(", impl):
        errors.append("TRECIP implementation is not identity-preserving")
    if re.search(r"TDIVS_IMPL[\s\S]{0,180}TRECIP\s*\(", impl):
        errors.append("TDIVS aliases TRECIP instead of preserving its own identity")
    if re.search(r"\bTileI32\s+mamulb\s*\(", low_level) or "cubeMamulb" in backend:
        errors.append("legacy mamulb identity remains in the Linx backend")
    if re.search(
        r"inline\s+void\s+THISTOGRAM\s*\([^,]+,[^,]+\)", tileop, re.MULTILINE
    ):
        errors.append("legacy two-operand THISTOGRAM convenience overload remains")
    if "dependent_false" not in generated or "defined(__LINXISA__)" not in generated:
        errors.append("canonical Linx dispatch does not fail closed for unsupported operations")

    for path in (ROOT / "include").rglob("*"):
        if not path.is_file() or "generated" in path.parts or path.suffix not in {".h", ".hpp", ".c", ".cc", ".cpp"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in DELETED:
            if re.search(rf"\b{name}(?:_IMPL)?\b", text):
                errors.append(f"{path.relative_to(ROOT)} retains deleted operation {name}")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("PTO 0.57.1 direct surface verified: exact 120 signatures, identities, and no D-ops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
