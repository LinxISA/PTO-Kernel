#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PTO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PTO_ROOT / "docs"
DEFAULT_PTO_ROOT = PTO_ROOT


def load_manifest():
    pto_root = Path(os.environ.get("PTO_KERNELS_ROOT", DEFAULT_PTO_ROOT)).expanduser().resolve()
    sys.path.insert(0, str(pto_root / "tools"))
    from benchmark_manifest import DEFAULT_WORKBOOK, build_manifest  # type: ignore
    from kernel_catalog import load_kernel_catalog  # type: ignore

    manifest = build_manifest(DEFAULT_WORKBOOK)
    return pto_root, DEFAULT_WORKBOOK, manifest, load_kernel_catalog()


def aggregate_kernels(manifest: dict, catalog: dict[str, str]) -> list[dict]:
    by_kernel: dict[str, dict] = {}
    for row in manifest["benchmarks"]:
        kernel = row.get("candidate_kernel") or ""
        if not kernel:
            continue
        agg = by_kernel.setdefault(
            kernel,
            {
                "kernel": kernel,
                "source_path": catalog.get(kernel, ""),
                "benchmarks": [],
                "status_counts": Counter(),
                "strategy_counts": Counter(),
                "op_kinds": set(),
                "deliverable_now_count": 0,
                "review_required_count": 0,
                "parity_covered": False,
                "kernel_style": row.get("kernel_style", ""),
                "xdim": row.get("xdim"),
                "ydim": row.get("ydim"),
                "tile_shape": row.get("tile_shape", ""),
            },
        )
        agg["benchmarks"].append(
            {
                "id": row["id"],
                "name": row["name"],
                "status": row["status"],
                "implementation_strategy": row["implementation_strategy"],
                "deliverable_now": row["deliverable_now"],
                "review_required": row["review_required"],
            }
        )
        agg["status_counts"][row["status"]] += 1
        agg["strategy_counts"][row["implementation_strategy"]] += 1
        agg["op_kinds"].add(row["op_kind"])
        if row["deliverable_now"]:
            agg["deliverable_now_count"] += 1
        if row["review_required"]:
            agg["review_required_count"] += 1
        agg["parity_covered"] = agg["parity_covered"] or bool(row["parity_covered"])

    kernels = []
    for agg in by_kernel.values():
        agg["benchmark_count"] = len(agg["benchmarks"])
        agg["benchmark_ids"] = [item["id"] for item in agg["benchmarks"]]
        agg["op_kinds"] = sorted(agg["op_kinds"])
        agg["status_counts"] = dict(sorted(agg["status_counts"].items()))
        agg["strategy_counts"] = dict(sorted(agg["strategy_counts"].items()))
        agg["primary_status"] = max(
            agg["status_counts"].items(),
            key=lambda item: (item[1], item[0]),
        )[0]
        kernels.append(agg)
    return sorted(kernels, key=lambda item: item["kernel"])


def aggregate_backlog(manifest: dict) -> list[dict]:
    backlog = []
    for row in manifest["benchmarks"]:
        if row.get("candidate_kernel"):
            continue
        backlog.append(
            {
                "id": row["id"],
                "name": row["name"],
                "status": row["status"],
                "op_kind": row["op_kind"],
                "implementation_strategy": row["implementation_strategy"],
                "deliverable_now": row["deliverable_now"],
                "review_required": row["review_required"],
                "implementation_note": row["implementation_note"],
                "input_shape": row["input_shape"],
                "output_shape": row["output_shape"],
            }
        )
    return backlog


def build_payload() -> dict:
    pto_root, workbook_path, manifest, catalog = load_manifest()
    summary = manifest["summary"]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": {
            "pto_kernels_root": str(pto_root),
            "workbook_path": str(workbook_path),
        },
        "summary": summary,
        "kernels": aggregate_kernels(manifest, catalog),
        "backlog": aggregate_backlog(manifest),
        "benchmarks": manifest["benchmarks"],
        "catalog": catalog,
    }
    return payload


def main() -> int:
    payload = build_payload()
    out_path = DOCS_ROOT / "data" / "status.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
