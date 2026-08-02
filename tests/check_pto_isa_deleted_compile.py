#!/usr/bin/env python3
"""Compile-negative gate for operations deleted by PTO 0.57.1."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


DELETED = ("TADDC", "TADDSC", "TFMA", "TFMOD", "TFMODS", "TLRELU", "TRANDOM", "TSUBC", "TSUBSC")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--include", type=Path, required=True)
    args = parser.parse_args()
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="pto-v0571-deleted-") as tmp:
        source = Path(tmp) / "deleted.cpp"
        for name in DELETED:
            source.write_text(
                "#include <pto/pto-inst.hpp>\n"
                f"int main() {{ pto::{name}(); }}\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [args.compiler, "-std=c++17", "-D__CPU_SIM=1", "-fsyntax-only",
                 f"-I{args.include}", str(source)],
                capture_output=True,
                text=True,
            )
            diagnostics = result.stdout + result.stderr
            if result.returncode == 0:
                errors.append(f"deleted operation still compiles: {name}")
            elif name not in diagnostics:
                errors.append(f"{name}: compile failed for an unrelated reason")
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("PTO 0.57.1 deleted operations rejected: " + ", ".join(DELETED))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
