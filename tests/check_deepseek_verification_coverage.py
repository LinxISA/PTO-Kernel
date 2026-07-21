#!/usr/bin/env python3
"""Audit DeepSeek kernel and kernel-reachable PTO intrinsic coverage."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


API_RE = re.compile(r"\b(deepseek_[a-z0-9_]+)\s*\(")
INTRINSIC_RE = re.compile(r"\bpto::(T[A-Z0-9_]+)\s*\(")
BSTART_RE = re.compile(r"\bBSTART\.(T[A-Z0-9.]+)\b")
ORACLE_RE = re.compile(
    r"//\s*PTO-ORACLE:\s*(deepseek_[a-z0-9_]+)\s*\|\s*([^\n]+)"
)


def manifest_apis(document: dict[str, object]) -> set[str]:
    apis: set[str] = set()
    families = document.get("families")
    if not isinstance(families, dict):
        return apis
    for family in families.values():
        if not isinstance(family, dict):
            continue
        ports = family.get("ports")
        if not isinstance(ports, dict):
            continue
        for names in ports.values():
            if isinstance(names, list):
                apis.update(str(name) for name in names)
    return apis


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def canonical_pto_name(name: str) -> str:
    if name == "TTRANSPOSE":
        return "TTRANS"
    return name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--super-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--asm", type=Path, nargs="*", default=[])
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    super_root = args.super_root.resolve()
    manifest_path = repo_root / "docs/upstream/deepseek-tilekernels.json"
    header_path = repo_root / "include/common/deepseek_tilekernels.hpp"
    helper_path = repo_root / "include/common/deepseek_tile_intrinsics.hpp"
    kernel_dir = repo_root / "kernels/upstream/deepseek"
    catalog_path = super_root / "isa/v0.57/state/pto_ops.json"
    avs_path = super_root / "avs/qemu/tests/17_deepseek_tilekernels.cpp"
    qemu_helper_path = super_root / "emulator/qemu/target/linx/helper.c"

    required_paths = [
        manifest_path,
        header_path,
        helper_path,
        catalog_path,
        avs_path,
        qemu_helper_path,
    ]
    errors = [f"missing required coverage input: {path}" for path in required_paths if not path.is_file()]
    kernel_paths = sorted(kernel_dir.glob("*.cpp"))
    if not kernel_paths:
        errors.append(f"no DeepSeek kernel sources found under {kernel_dir}")
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    manifest = json.loads(read_text(manifest_path))
    catalog = json.loads(read_text(catalog_path))
    expected_apis = manifest_apis(manifest)
    declared_apis = set(API_RE.findall(read_text(header_path)))
    kernel_text = "\n".join(read_text(path) for path in kernel_paths)
    defined_apis = set(API_RE.findall(kernel_text))
    avs_text = read_text(avs_path)
    avs_apis = set(API_RE.findall(avs_text))
    oracle_entries = ORACLE_RE.findall(avs_text)
    oracle_apis = {name for name, _ in oracle_entries}

    catalog_names = {
        str(operation.get("name"))
        for operation in catalog.get("operations", [])
        if isinstance(operation, dict)
    }
    source_text = kernel_text + "\n" + read_text(helper_path)
    reachable_intrinsics = set(INTRINSIC_RE.findall(source_text)) | {"TLOAD", "TSTORE"}
    reachable_catalog_names = {canonical_pto_name(name) for name in reachable_intrinsics}

    missing_declarations = sorted(expected_apis - declared_apis)
    missing_definitions = sorted(expected_apis - defined_apis)
    missing_qemu_invocations = sorted(expected_apis - avs_apis)
    extra_declarations = sorted(declared_apis - expected_apis)
    extra_definitions = sorted(defined_apis - expected_apis)
    unknown_intrinsics = sorted(reachable_catalog_names - catalog_names)
    missing_oracles = sorted(expected_apis - oracle_apis)
    extra_oracles = sorted(oracle_apis - expected_apis)

    operation_by_name = {
        str(operation.get("name")): operation
        for operation in catalog.get("operations", [])
        if isinstance(operation, dict)
    }
    qemu_helper_text = read_text(qemu_helper_path).lower()
    qemu_semantic_names: set[str] = set()
    for name in reachable_catalog_names:
        operation = operation_by_name.get(name, {})
        disposition = operation.get("disposition", {})
        family = disposition.get("family") if isinstance(disposition, dict) else None
        selector = disposition.get("selector") if isinstance(disposition, dict) else None
        if family == "TEPL" and isinstance(selector, str):
            selector_token = f"0x{int(selector, 16):03x}u"
            if selector_token in qemu_helper_text:
                qemu_semantic_names.add(name)
        elif family == "TMA" and f"linx_tma_{name.lower()}" in qemu_helper_text:
            qemu_semantic_names.add(name)
    missing_qemu_semantics = sorted(reachable_catalog_names - qemu_semantic_names)

    for label, values in (
        ("manifest APIs missing public declarations", missing_declarations),
        ("manifest APIs missing implementations", missing_definitions),
        ("manifest APIs missing QEMU AVS invocation", missing_qemu_invocations),
        ("public declarations absent from manifest", extra_declarations),
        ("implementations absent from manifest", extra_definitions),
        ("kernel-reachable intrinsics absent from v0.57 catalog", unknown_intrinsics),
        ("manifest APIs missing runtime oracle annotations", missing_oracles),
        ("runtime oracle annotations absent from manifest", extra_oracles),
        ("kernel-reachable intrinsics missing QEMU semantic cases", missing_qemu_semantics),
    ):
        if values:
            errors.append(f"{label}: {', '.join(values)}")

    emitted_intrinsics: set[str] = set()
    if args.asm:
        missing_asm = [path for path in args.asm if not path.is_file()]
        if missing_asm:
            errors.extend(f"missing assembly coverage input: {path}" for path in missing_asm)
        else:
            for path in args.asm:
                emitted_intrinsics.update(BSTART_RE.findall(read_text(path)))
            missing_emission = sorted(reachable_intrinsics - emitted_intrinsics)
            if missing_emission:
                errors.append(
                    "kernel-reachable intrinsics missing from Linx assembly: "
                    + ", ".join(missing_emission)
                )

    report = {
        "schema_version": 1,
        "catalog": {
            "operation_count": len(catalog_names),
            "expected_operation_count": 111,
        },
        "kernel_api": {
            "manifest_count": len(expected_apis),
            "declared_count": len(expected_apis & declared_apis),
            "implemented_count": len(expected_apis & defined_apis),
            "qemu_invoked_count": len(expected_apis & avs_apis),
            "runtime_oracle_count": len(expected_apis & oracle_apis),
            "missing_declarations": missing_declarations,
            "missing_definitions": missing_definitions,
            "missing_qemu_invocations": missing_qemu_invocations,
            "missing_runtime_oracles": missing_oracles,
        },
        "kernel_reachable_intrinsics": {
            "count": len(reachable_intrinsics),
            "names": sorted(reachable_intrinsics),
            "catalog_mapped_count": len(reachable_catalog_names & catalog_names),
            "qemu_semantic_count": len(reachable_catalog_names & qemu_semantic_names),
            "qemu_semantic_names": sorted(qemu_semantic_names),
            "missing_qemu_semantics": missing_qemu_semantics,
            "emitted_count": len(reachable_intrinsics & emitted_intrinsics) if args.asm else None,
            "emitted_names": sorted(reachable_intrinsics & emitted_intrinsics) if args.asm else [],
        },
        "errors": errors,
    }
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        "OK: "
        f"{len(expected_apis)}/{len(expected_apis)} APIs declared, implemented, QEMU-invoked, and oracle-annotated; "
        f"{len(reachable_intrinsics)}/{len(reachable_intrinsics)} kernel-reachable PTO intrinsics "
        + ("cataloged, QEMU-mapped, and emitted" if args.asm else "cataloged and QEMU-mapped")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
