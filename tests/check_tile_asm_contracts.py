#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from pathlib import Path

REQUIRED_PATTERNS = [
    re.compile(r"\\bBSTART\\.T(LOAD|STORE|MATMUL|MATMUL\\.ACC)\\b"),
]
FORBIDDEN_PATTERNS = [
    re.compile(r"(^|[^A-Za-z0-9_])L\\."),
    re.compile(r"\\b(set_flag|wait_flag|B\\.SET|B\\.WAIT)\\b"),
]


def check_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    errors: list[str] = []
    if not any(p.search(text) for p in REQUIRED_PATTERNS):
        errors.append(f"missing required BSTART.T* marker: {path}")
    for pat in FORBIDDEN_PATTERNS:
        if pat.search(text):
            errors.append(f"forbidden pattern '{pat.pattern}' found: {path}")
    return errors


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: check_tile_asm_contracts.py <asm> [<asm> ...]", file=sys.stderr)
        return 2

    errors: list[str] = []
    for item in argv[1:]:
        p = Path(item)
        if not p.exists():
            errors.append(f"missing asm file: {p}")
            continue
        errors.extend(check_file(p))

    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        return 1
    print("OK: asm contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
