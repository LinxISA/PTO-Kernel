#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PTO_ROOT = SCRIPT.parents[1]
sys.path.insert(0, str(PTO_ROOT / "tools"))

from benchmark_manifest import (  # noqa: E402
    DEFAULT_WORKBOOK,
    PARITY_ROW_TO_KERNEL,
    PARITY_KERNEL_ORDER,
    build_manifest,
    load_workbook_tables,
    local_kernel_names,
)


def require(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"error: {message}")


def main() -> int:
    if not DEFAULT_WORKBOOK.exists():
        print(f"SKIP: workbook not found at {DEFAULT_WORKBOOK}")
        return 0

    manifest = build_manifest(DEFAULT_WORKBOOK)
    summary = manifest["summary"]
    kernels = local_kernel_names()
    tables = load_workbook_tables(DEFAULT_WORKBOOK)

    require(summary["benchmark_count"] == 51, "expected 51 benchmark rows")
    require(summary["deliverable_now_count"] == 42, "expected 42 deliverable benchmark rows")
    require(summary["review_required_count"] == 17, "expected 17 review-required rows from Review_Items")
    require(summary["parity_covered_count"] == len(PARITY_ROW_TO_KERNEL), "unexpected parity benchmark count")

    require(summary["status_counts"] == {
        "confirmed": 25,
        "confirmed_review": 7,
        "need_verify": 9,
        "proposed_high": 9,
        "proposed_medium": 1,
    }, "status counts drifted from workbook")
    require(summary["implementation_strategy_counts"] == {
        "blocked_review": 7,
        "new_kernel": 16,
        "reuse_existing": 13,
        "specialize_existing": 15,
    }, "implementation strategy counts drifted")

    review_ids = {row["ID"] for row in tables["Review_Items"]}
    expected_review = {
        item["id"] for item in manifest["benchmarks"]
        if item["review_required"]
    }
    require(review_ids == expected_review, "Review_Items sheet no longer matches review-required benchmark ids")

    parity_rows = {}
    for item in manifest["benchmarks"]:
        require(item["id"] not in parity_rows, f"duplicate benchmark id in manifest: {item['id']}")
        parity_rows[item["id"]] = item
        require(all(kernel in kernels for kernel in item["local_kernels"]),
                f"missing local kernel mapping for {item['id']}: {item['local_kernels']}")
        if item["parity_kernel"]:
            require(item["parity_kernel"] in PARITY_KERNEL_ORDER,
                    f"parity kernel not registered in order list: {item['parity_kernel']}")

    for bid, kernel in PARITY_ROW_TO_KERNEL.items():
        require(parity_rows[bid]["parity_kernel"] == kernel,
                f"unexpected parity kernel mapping for {bid}")

    require(parity_rows["B015"]["implementation_strategy"] == "reuse_existing",
            "FlashMLA row should remain attached to the existing local kernel")
    require(parity_rows["B015"]["review_required"],
            "FlashMLA row should still require review before delivery")
    require(parity_rows["B017"]["implementation_strategy"] == "reuse_existing",
            "SFA row should remain mapped to the local sparse-attention kernel")
    require(parity_rows["B022"]["implementation_strategy"] == "specialize_existing",
            "init_routing row should be backed by existing MoE primitives")
    require(parity_rows["B047"]["implementation_strategy"] == "new_kernel",
            "div benchmark should remain a new-kernel item")
    require(parity_rows["B014"]["kernel_style"] == "mixed_tile_simt",
            "flash attention pilot row should advertise mixed_tile_simt style")
    require(parity_rows["B016"]["kernel_style"] == "mixed_tile_simt",
            "gqa pilot row should advertise mixed_tile_simt style")
    require(parity_rows["B017"]["kernel_style"] == "mixed_tile_simt",
            "sparse attention pilot row should advertise mixed_tile_simt style")
    require(parity_rows["B030"]["kernel_style"] == "mixed_tile_simt",
            "rmsnorm pilot row should advertise mixed_tile_simt style")
    require(parity_rows["B014"]["ydim"] == 2,
            "flash attention pilot row should record Ydim=2")
    require(parity_rows["B030"]["tile_shape"] == "1x16",
            "rmsnorm pilot row should record its vec-tile shape")

    print("OK: benchmark manifest matches workbook counts and kernel mappings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
