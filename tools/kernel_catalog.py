#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).resolve()
PTO_ROOT = SCRIPT.parents[1]
KERNEL_ROOT = PTO_ROOT / "kernels"
CATALOG_PATH = KERNEL_ROOT / "catalog.txt"


def load_kernel_catalog() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_line in CATALOG_PATH.read_text(encoding="utf-8").splitlines():
        entry = raw_line.strip()
        if not entry or entry.startswith("#"):
            continue
        name = Path(entry).stem
        if name in mapping:
            raise RuntimeError(f"duplicate kernel name in catalog: {name}")
        mapping[name] = entry
    return mapping


def local_kernel_names() -> set[str]:
    return set(load_kernel_catalog())
