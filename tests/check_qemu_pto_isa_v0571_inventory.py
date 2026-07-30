#!/usr/bin/env python3
"""Compare QEMU's executable tile inventory exactly with pto-spec 0.57.1."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


CASE_RE = re.compile(r"case\s+(0x[0-9a-fA-F]+)u?\s*:\s*/\*\s*([A-Z][A-Z0-9_]*)\s*\*/")
ENUM_RE = re.compile(r"LINX_(TMA|CUBE)_([A-Z][A-Z0-9_]*)\s*=\s*(0x[0-9a-fA-F]+|[0-9]+)")


def parse_inventory(helper: str) -> dict[str, dict[str, int]]:
    start = helper.find("static bool linx_tile_tepl_selector_executable")
    end = helper.find("#define LINX_TILE_DTYPE_MASK", start)
    tepl_text = helper[start:end] if start >= 0 and end > start else ""
    inventories: dict[str, dict[str, int]] = {"TEPL": {}, "TMA": {}, "CUBE": {}}
    for value, name in CASE_RE.findall(tepl_text):
        inventories["TEPL"][name] = int(value, 16)
    for family, anchor in (("TMA", "LINX_TMA_TLOAD"), ("CUBE", "LINX_CUBE_")):
        anchor_at = helper.find(anchor)
        enum_start = helper.rfind("enum {", 0, anchor_at)
        enum_end = helper.find("};", anchor_at)
        enum_text = helper[enum_start:enum_end] if enum_start >= 0 and enum_end > anchor_at else ""
        for parsed_family, name, value in ENUM_RE.findall(enum_text):
            if parsed_family == family:
                inventories[family][name] = int(value, 0)
    return inventories


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--qemu-helper", type=Path, required=True)
    args = parser.parse_args()
    if not args.catalog.is_file() or not args.qemu_helper.is_file():
        print("error: catalog and QEMU helper must both exist", file=sys.stderr)
        return 1
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    expected: dict[str, dict[str, int]] = {"TEPL": {}, "TMA": {}, "CUBE": {}}
    for item in catalog["operations"]:
        family = str(item["family"])
        expected[family][str(item["name"])] = (
            int(str(item["selector"]), 16) if family == "TEPL" else int(item["function"])
        )
    actual = parse_inventory(args.qemu_helper.read_text(encoding="utf-8", errors="replace"))
    errors: list[str] = []
    for family in ("TEPL", "TMA", "CUBE"):
        missing = sorted(set(expected[family]) - set(actual[family]))
        extra = sorted(set(actual[family]) - set(expected[family]))
        mismatched = sorted(
            name for name in set(expected[family]) & set(actual[family])
            if expected[family][name] != actual[family][name]
        )
        if missing:
            errors.append(f"QEMU {family} inventory missing: {', '.join(missing)}")
        if extra:
            errors.append(f"QEMU {family} inventory has non-0.57.1 identities: {', '.join(extra)}")
        for name in mismatched:
            errors.append(
                f"QEMU {family} {name}: expected {expected[family][name]:#x}, "
                f"got {actual[family][name]:#x}"
            )
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("QEMU PTO 0.57.1 inventory exact: 98 TEPL, 9 TMA, 13 CUBE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
