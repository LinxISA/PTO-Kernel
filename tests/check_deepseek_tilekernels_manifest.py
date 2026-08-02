#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "upstream" / "deepseek-tilekernels.json"
PUBLIC_HEADER = ROOT / "include" / "common" / "deepseek_tilekernels.hpp"
KERNEL_DIR = ROOT / "kernels" / "upstream" / "deepseek"

EXPECTED_FAMILIES = {
    "engram": 4,
    "mhc": 9,
    "moe": 11,
    "quant": 12,
    "transpose": 1,
}


def main() -> int:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    errors: list[str] = []
    upstream = data.get("upstream", {})
    commit = upstream.get("commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        errors.append("upstream.commit must be a full 40-character SHA")
    if upstream.get("repository") != "https://github.com/deepseek-ai/TileKernels":
        errors.append("unexpected upstream repository")

    header = PUBLIC_HEADER.read_text(encoding="utf-8")
    families = data.get("families", {})
    mapped_sources = 0
    mapped_apis: set[str] = set()
    for family, expected_count in EXPECTED_FAMILIES.items():
        entry = families.get(family)
        if not isinstance(entry, dict):
            errors.append(f"missing family: {family}")
            continue
        sources = entry.get("sources", [])
        ports = entry.get("ports", {})
        if len(sources) != expected_count:
            errors.append(
                f"{family}: expected {expected_count} upstream sources, "
                f"found {len(sources)}"
            )
        if entry.get("status") != "implemented":
            errors.append(f"{family}: status is not implemented")
        if set(sources) != set(ports):
            errors.append(f"{family}: sources and ports keys differ")
        source_file = KERNEL_DIR / f"{family}.cpp"
        if not source_file.is_file():
            errors.append(f"{family}: missing PTO source {source_file}")
            continue
        implementation = source_file.read_text(encoding="utf-8")
        if re.search(r"\b(for|while|do)\s*(?:\(|\{)", implementation):
            errors.append(f"{family}: scalar loop found in PTO kernel source")
        if "extended_kernel_runtime" in implementation:
            errors.append(f"{family}: scalar runtime helper included by PTO kernel")
        if "pto::T" not in implementation:
            errors.append(f"{family}: no named PTO ISA intrinsic calls found")
        if "pto::THISTOGRAM" in implementation:
            errors.append(
                f"{family}: global histogram algorithm must not misuse row-wise THISTOGRAM"
            )
        if "pto::TSORT" in implementation:
            errors.append(
                f"{family}: algorithm without an index destination must not misuse TSORT"
            )
        for source in sources:
            apis = ports.get(source, [])
            if not apis:
                errors.append(f"{family}/{source}: no mapped PTO API")
            mapped_sources += 1
            for api in apis:
                if api in mapped_apis:
                    errors.append(f"duplicate mapped PTO API: {api}")
                mapped_apis.add(api)
                if api not in header:
                    errors.append(f"{family}/{source}: API missing from header: {api}")
                if api not in implementation:
                    errors.append(
                        f"{family}/{source}: API missing from implementation: {api}"
                    )

    if mapped_sources != 37:
        errors.append(f"expected 37 mapped upstream kernel sources, found {mapped_sources}")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"OK: mapped {mapped_sources} upstream kernel sources to "
        f"{len(mapped_apis)} scalar-data-loop-free PTO APIs at {commit}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
