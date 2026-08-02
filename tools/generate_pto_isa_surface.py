#!/usr/bin/env python3
"""Generate the PTO-Kernel direct ISA identity surface from pto-spec.

The checked-in JSON projection is a reviewable cache, not an independent
source of truth.  Passing --pto-spec-root verifies the repository revision and
catalog digest recorded by docs/contracts/pto_isa_v0571.lock.json before any
output is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "docs/contracts/pto_isa_v0571.lock.json"
SURFACE_PATH = ROOT / "docs/contracts/generated/pto_isa_v0571_surface.json"
HEADER_PATH = ROOT / "include/common/generated/pto_isa_v0571.hpp"
API_PATH = ROOT / "include/pto/common/generated/pto_isa_v0571_api.inc"
API_COMPILE_PATH = ROOT / "tests/generated/pto_isa_v0571_calls.inc"

EXPECTED_DELETED = {
    "TADDC",
    "TADDSC",
    "TFMA",
    "TFMOD",
    "TFMODS",
    "TLRELU",
    "TRANDOM",
    "TSUBC",
    "TSUBSC",
}

# Canonical signatures whose Linx implementation is already exact.  Every
# other operation deliberately reaches the generated dependent static_assert;
# adding an entry here requires an exact `_IMPL` with the catalog operand order.
LINX_IMPLEMENTED = {
    "TADD", "TSUB", "TMUL", "TMAX", "TEXP", "TEXPANDS", "TRECIP",
    "TMULS", "TDIVS", "TLOAD", "TSTORE",
}


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(document: dict[str, object]) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def source_catalog(lock: dict[str, object], pto_spec_root: Path) -> dict[str, object]:
    source = lock["source"]
    assert isinstance(source, dict)
    expected_commit = str(source["commit"])
    actual_commit = checked_git_head(pto_spec_root)
    if actual_commit != expected_commit:
        raise ValueError(
            f"pto-spec revision mismatch: expected {expected_commit}, got {actual_commit}"
        )
    def check_file(path_key: str, sha_key: str) -> Path:
        path = pto_spec_root / str(source[path_key])
        actual_sha = sha256(path)
        expected_sha = str(source[sha_key])
        if actual_sha != expected_sha:
            raise ValueError(
                f"pto-spec {path_key} digest mismatch: expected {expected_sha}, got {actual_sha}"
            )
        return path

    catalog_path = check_file("catalog", "catalog_sha256")
    manifest = None
    if "release_manifest" in source:
        manifest_path = check_file("release_manifest", "release_manifest_sha256")
        manifest = read_json(manifest_path)
        expected_content = str(source.get("content_sha256", ""))
        if manifest.get("content_sha256") != expected_content:
            raise ValueError(
                "pto-spec content digest mismatch: "
                f"expected {expected_content}, got {manifest.get('content_sha256')}"
            )
        expected_encoding = str(source.get("encoding_projection_sha256", ""))
        if manifest.get("encoding_projection_sha256") != expected_encoding:
            raise ValueError(
                "pto-spec encoding projection digest mismatch: "
                f"expected {expected_encoding}, got {manifest.get('encoding_projection_sha256')}"
            )
    hardware = None
    if "hardware_conformance_profile" in source:
        hardware_path = check_file(
            "hardware_conformance_profile",
            "hardware_conformance_profile_sha256",
        )
        hardware = read_json(hardware_path)
        declared = (manifest or {}).get("hardware_conformance_profile", {})
        if (
            declared.get("path") != source["hardware_conformance_profile"]
            or declared.get("sha256")
            != source["hardware_conformance_profile_sha256"]
            or declared.get("profile_id") != hardware.get("profile_id")
        ):
            raise ValueError("pto-spec manifest does not bind the locked hardware profile")
    if "numeric_vectors" in source:
        vectors_path = check_file("numeric_vectors", "numeric_vectors_sha256")
        vectors = read_json(vectors_path)
        if (
            hardware is None
            or vectors.get("hardware_profile_id") != hardware.get("profile_id")
            or vectors.get("hardware_profile_sha256")
            != source["hardware_conformance_profile_sha256"]
        ):
            raise ValueError("pto-spec numeric vectors do not bind the locked hardware profile")
    return read_json(catalog_path)


def project_catalog(lock: dict[str, object], catalog: dict[str, object]) -> dict[str, object]:
    operations = catalog.get("operations")
    if not isinstance(operations, list):
        raise ValueError("pto-spec catalog operations must be a list")

    projected: list[dict[str, object]] = []
    for raw in operations:
        if not isinstance(raw, dict):
            raise ValueError("pto-spec operation entries must be objects")
        name = str(raw["name"])
        family = str(raw["family"])
        function = int(raw["function"])
        item: dict[str, object] = {
            "name": name,
            "family": family,
            "command_mnemonic": str(raw["command_mnemonic"]),
            "function": function,
            "operands": [
                {"field": str(operand["field"]), "role": str(operand["role"])}
                for operand in raw.get("operands", [])
            ],
        }
        if family == "TEPL":
            mode = int(raw["mode"])
            selector = int(str(raw["selector"]), 16)
            if selector != (mode << 5) | function:
                raise ValueError(
                    f"{name}: selector {selector:#05x} != Mode/Function "
                    f"packing ({mode} << 5) | {function}"
                )
            item["mode"] = mode
            item["selector"] = f"0x{selector:03X}"
        projected.append(item)

    names = [str(item["name"]) for item in projected]
    if len(names) != len(set(names)):
        raise ValueError("pto-spec direct operation names must be unique")
    expected_count = int(lock["operation_count"])
    if len(names) != expected_count or int(catalog.get("operation_count", -1)) != expected_count:
        raise ValueError(f"pto-spec catalog must contain exactly {expected_count} operations")

    counts = Counter(str(item["family"]) for item in projected)
    expected_counts = {
        str(key): int(value)
        for key, value in dict(lock["family_counts"]).items()
    }
    if dict(counts) != expected_counts:
        raise ValueError(f"family counts differ: expected {expected_counts}, got {dict(counts)}")

    deleted = {str(name) for name in catalog.get("deleted_names", [])}
    if deleted != EXPECTED_DELETED:
        raise ValueError(
            f"deleted-name set differs: expected {sorted(EXPECTED_DELETED)}, got {sorted(deleted)}"
        )
    overlap = deleted & set(names)
    if overlap:
        raise ValueError(f"deleted operations remain active: {sorted(overlap)}")

    for family in ("TMA", "CUBE"):
        functions = [
            int(item["function"])
            for item in projected
            if item["family"] == family
        ]
        if len(functions) != len(set(functions)):
            raise ValueError(f"{family} function selectors must be unique")
        if family == "TMA" and set(functions) != set(range(9)):
            raise ValueError(f"TMA function selectors must be dense 0..8, got {sorted(functions)}")

    source = lock["source"]
    assert isinstance(source, dict)
    return {
        "schema_version": 1,
        "profile": str(lock["profile"]),
        "source": dict(source),
        "operation_count": expected_count,
        "family_counts": expected_counts,
        "deleted_names": sorted(deleted),
        "operations": projected,
    }


def validate_projection(lock: dict[str, object], projection: dict[str, object]) -> None:
    source = lock["source"]
    assert isinstance(source, dict)
    if projection.get("profile") != lock.get("profile"):
        raise ValueError("checked-in surface profile differs from lock")
    if projection.get("operation_count") != lock.get("operation_count"):
        raise ValueError("checked-in surface operation count differs from lock")
    if projection.get("family_counts") != lock.get("family_counts"):
        raise ValueError("checked-in surface family counts differ from lock")
    projection_source = projection.get("source")
    if projection_source != source:
        raise ValueError("checked-in surface source metadata differs from lock")
    operations = projection.get("operations")
    if not isinstance(operations, list):
        raise ValueError("checked-in surface operations must be a list")
    names = [str(item["name"]) for item in operations if isinstance(item, dict)]
    if len(names) != int(lock["operation_count"]) or len(names) != len(set(names)):
        raise ValueError("checked-in surface must contain 120 unique operations")
    if set(projection.get("deleted_names", [])) != EXPECTED_DELETED:
        raise ValueError("checked-in surface deleted-name set differs from 0.57.1")
    if set(names) & EXPECTED_DELETED:
        raise ValueError("checked-in surface includes a deleted operation")


def cpp_family(family: str) -> str:
    return {"TEPL": "Family::TEPL", "TMA": "Family::TMA", "CUBE": "Family::CUBE"}[family]


def render_header(projection: dict[str, object]) -> str:
    operations = projection["operations"]
    assert isinstance(operations, list)
    source = projection["source"]
    assert isinstance(source, dict)
    lines = [
        "// Generated by tools/generate_pto_isa_surface.py. DO NOT EDIT.",
        f"// Source: {source['repository']}@{source['commit']}",
        "#ifndef PTO_COMMON_GENERATED_PTO_ISA_V0571_HPP",
        "#define PTO_COMMON_GENERATED_PTO_ISA_V0571_HPP",
        "",
        "namespace pto {",
        "namespace isa_v0571 {",
        "",
        "enum class Family : unsigned { TEPL, TMA, CUBE };",
        "",
        "constexpr unsigned kOperationCount = 120u;",
        "constexpr unsigned kTeplOperationCount = 98u;",
        "constexpr unsigned kTmaOperationCount = 9u;",
        "constexpr unsigned kCubeOperationCount = 13u;",
        "",
        "constexpr unsigned packTeplSelector(unsigned mode, unsigned function) {",
        "  return (mode << 5u) | function;",
        "}",
        "",
        "struct OperationDescriptor {",
        "  const char *name;",
        "  Family family;",
        "  const char *commandMnemonic;",
        "  unsigned mode;",
        "  unsigned function;",
        "  unsigned selector;",
        "};",
        "",
    ]

    for family in ("TEPL", "TMA", "CUBE"):
        lines.append(f"namespace {family.lower()} {{")
        for item in operations:
            assert isinstance(item, dict)
            if item["family"] != family:
                continue
            name = str(item["name"])
            function = int(item["function"])
            if family == "TEPL":
                mode = int(item["mode"])
                selector = int(str(item["selector"]), 16)
                lines.extend(
                    [
                        f"constexpr unsigned {name}_MODE = {mode}u;",
                        f"constexpr unsigned {name}_FUNCTION = {function}u;",
                        f"constexpr unsigned {name} = packTeplSelector({name}_MODE, {name}_FUNCTION);",
                        f"static_assert({name} == 0x{selector:03X}u, \"{name} raw selector parity\");",
                    ]
                )
            else:
                command = str(item["command_mnemonic"])
                lines.extend(
                    [
                        f"constexpr unsigned {name}_FUNCTION = {function}u;",
                        f"constexpr const char {name}_COMMAND[] = \"{command}\";",
                    ]
                )
        lines.extend([f"}} // namespace {family.lower()}", ""])

    lines.append("constexpr OperationDescriptor kDirectOperations[] = {")
    for item in operations:
        assert isinstance(item, dict)
        family = str(item["family"])
        mode = int(item.get("mode", 0))
        function = int(item["function"])
        selector = int(str(item.get("selector", "0")), 16)
        lines.append(
            f'  {{"{item["name"]}", {cpp_family(family)}, '
            f'"{item["command_mnemonic"]}", {mode}u, {function}u, 0x{selector:03X}u}},'
        )
    lines.extend(
        [
            "};",
            "static_assert(sizeof(kDirectOperations) / sizeof(kDirectOperations[0]) ==",
            "                  kOperationCount, \"PTO 0.57.1 direct surface count\");",
            "",
            "} // namespace isa_v0571",
            "} // namespace pto",
            "",
            "#endif // PTO_COMMON_GENERATED_PTO_ISA_V0571_HPP",
            "",
        ]
    )
    return "\n".join(lines)


def render_api_bridge(projection: dict[str, object]) -> str:
    """Render the complete, exact-arity PTO 0.57.1 public API.

    The catalog is authoritative for both operand order and arity. Hand-written
    compatibility or convenience entry points must never suppress or overload
    one of these generated direct operations.
    """
    operations = projection["operations"]
    assert isinstance(operations, list)
    source = projection["source"]
    assert isinstance(source, dict)
    lines = [
        "// Generated by tools/generate_pto_isa_surface.py. DO NOT EDIT.",
        f"// Source: {source['repository']}@{source['commit']}",
        "// Included inside namespace pto by pto_instr.hpp.",
        "",
        "namespace isa_v0571_detail {",
        "template <typename...> struct dependent_false { static constexpr bool value = false; };",
        "",
        "template <typename Operation, typename... Operands>",
        "PTO_INST RecordEvent dispatch(Operands &...operands) {",
        "#if defined(__LINXISA__)",
        "  (void)sizeof...(operands);",
        "  static_assert(dependent_false<Operation, Operands...>::value,",
        "                \"PTO Linx strict-0.57.1: unsupported canonical direct operation\");",
        "#else",
        "  (void)sizeof...(operands);",
        "#endif",
        "  return {};",
        "}",
        "} // namespace isa_v0571_detail",
        "",
    ]
    for item in operations:
        assert isinstance(item, dict)
        name = str(item["name"])
        operands = item.get("operands", [])
        assert isinstance(operands, list)
        template_types = [f"typename Operand{index}" for index in range(len(operands))]
        parameters: list[str] = []
        argument_names: list[str] = []
        for index, operand in enumerate(operands):
            assert isinstance(operand, dict)
            field = str(operand["field"])
            argument_names.append(field)
            reference = field.startswith(("destination", "source", "address"))
            parameters.append(f"Operand{index} {'&' if reference else ''}{field}")
        forwarded = ", ".join(argument_names)
        tag = f"{name}Operation"
        if name in LINX_IMPLEMENTED:
            implementation = [
                "#if defined(__LINXISA__)",
                f"  MAP_INSTR_IMPL({name}{', ' if forwarded else ''}{forwarded});",
                "  return {};",
                "#else",
                f"  return isa_v0571_detail::dispatch<{tag}>({forwarded});",
                "#endif",
            ]
        else:
            implementation = [
                f"  return isa_v0571_detail::dispatch<{tag}>({forwarded});",
            ]
        lines.extend(
            [
                f"// PTO-ISA-API: {name}({', '.join(argument_names)})",
                f"struct {tag} {{}};",
                f"template <{', '.join(template_types)}>",
                f"PTO_INST RecordEvent {name}({', '.join(parameters)}) {{",
                *implementation,
                "}",
                "",
            ]
        )
    return "\n".join(lines)


def render_api_compile_calls(projection: dict[str, object]) -> str:
    operations = projection["operations"]
    assert isinstance(operations, list)
    lines = [
        "// Generated by tools/generate_pto_isa_surface.py. DO NOT EDIT.",
        "// Every canonical operation is instantiated by the host compile gate.",
    ]
    for item in operations:
        assert isinstance(item, dict)
        operands = item.get("operands", [])
        assert isinstance(operands, list)
        arguments = ", ".join("operand" for _ in operands)
        lines.append(f"  (void)pto::{item['name']}({arguments});")
    lines.append("")
    return "\n".join(lines)


def compare(path: Path, expected: str, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing generated artifact: {path.relative_to(ROOT)}")
        return
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        errors.append(f"stale generated artifact: {path.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pto-spec-root", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.write == args.check:
        parser.error("choose exactly one of --write or --check")

    lock = read_json(LOCK_PATH)
    if args.pto_spec_root:
        catalog = source_catalog(lock, args.pto_spec_root.resolve())
        projection = project_catalog(lock, catalog)
    else:
        if args.write:
            parser.error("--write requires --pto-spec-root")
        projection = read_json(SURFACE_PATH)
        validate_projection(lock, projection)

    surface_text = canonical_json(projection)
    header_text = render_header(projection)
    api_text = render_api_bridge(projection)
    api_compile_text = render_api_compile_calls(projection)
    if args.write:
        SURFACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        HEADER_PATH.parent.mkdir(parents=True, exist_ok=True)
        API_PATH.parent.mkdir(parents=True, exist_ok=True)
        API_COMPILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SURFACE_PATH.write_text(surface_text, encoding="utf-8")
        HEADER_PATH.write_text(header_text, encoding="utf-8")
        API_PATH.write_text(api_text, encoding="utf-8")
        API_COMPILE_PATH.write_text(api_compile_text, encoding="utf-8")
        print("generated PTO 0.57.1 direct surface: 120 operations")
        return 0

    errors: list[str] = []
    compare(SURFACE_PATH, surface_text, errors)
    compare(HEADER_PATH, header_text, errors)
    compare(API_PATH, api_text, errors)
    compare(API_COMPILE_PATH, api_compile_text, errors)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("PTO 0.57.1 generated surface clean: 120 operations (98 TEPL, 9 TMA, 13 CUBE)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
