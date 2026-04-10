#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
PTO_ROOT = SCRIPT.parents[1]
KERNEL_DIR = PTO_ROOT / "kernels"
DEFAULT_WORKBOOK = Path("~/Documents/benchmark_master_list_completed.xlsx").expanduser()
XML_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

PARITY_KERNEL_ORDER = [
    "tload_store",
    "mamulb",
    "tmatmul_acc",
    "gemm",
    "gemm_basic",
    "gemm_demo",
    "gemm_performance",
    "add_custom",
    "flash_attention",
    "flash_attention_demo",
    "flash_attention_masked",
    "fa_performance",
    "mla_attention_demo",
    "flash_attention_cube_fp16",
    "flash_attention_vec_fp32",
    "flash_attention_vec_fp16",
    "gqa_fp16",
    "sparse_attention_local_fp16",
    "rmsnorm_fp16",
    "gelu_fp32",
    "argmax_fp32",
    "gather_fp32",
    "concat_fp32",
    "scatter_fp32",
    "permute_nhwc_nchw_fp32",
    "transpose_large_fp32",
    "unsorted_segment_sum_fp32",
    "unique_i32",
]

REUSE_EXISTING: dict[str, tuple[list[str], str]] = {
    "B011": (["gelu_fp32"], "Reuse the existing GELU kernel through a bf16 benchmark adapter."),
    "B014": (["flash_attention_cube_fp16"], "Reuse the causal cube-backed flash-attention kernel."),
    "B015": (["flash_mla_deepseekv3_fp16"], "Reuse the existing FlashMLA kernel once its benchmark wrapper is added."),
    "B016": (["gqa_fp16"], "Reuse the grouped-query attention kernel."),
    "B017": (["sparse_attention_local_fp16"], "Reuse the local sparse-attention kernel once SFA naming is confirmed."),
    "B030": (["rmsnorm_fp16"], "Reuse the RMSNorm kernel through a bf16 benchmark adapter."),
    "B039": (["gather_fp32"], "Reuse the generic gather kernel for the recommendation case."),
    "B040": (["gather_fp32"], "Reuse the generic gather kernel for the small-table case."),
    "B041": (["gather_fp32"], "Reuse the generic gather kernel for the Lightning Indexer gather stage."),
    "B042": (["concat_fp32"], "Reuse the concat kernel with a row-specific shape adapter."),
    "B043": (["scatter_fp32"], "Reuse the scatter kernel with a row-specific shape adapter."),
    "B044": (["unsorted_segment_sum_fp32"], "Reuse the unsorted-segment-sum kernel."),
    "B051": (["unique_i32"], "Reuse the unique kernel for the int32 duplicate-elimination case."),
}

SPECIALIZE_EXISTING: dict[str, tuple[list[str], str]] = {
    "B001": (["gemm"], "Add a benchmark adapter that specializes the existing GEMM path to the DeepSeekV3 train shape."),
    "B002": (["gemm"], "Add a benchmark adapter that preserves the non-128B-aligned GEMM stress shape."),
    "B003": ([], "Add a dedicated quantized matmul specialization for the HIF4 decode case."),
    "B004": ([], "Add a dedicated quantized matmul specialization for the HIF4 prefill/train case."),
    "B005": (["matmul_a16w4"], "Specialize the existing A16W4 path for the grouped-scale benchmark row."),
    "B006": (["gemm"], "Split the backward matmul benchmark into dx and dw adapters on top of the GEMM family."),
    "B009": ([], "Add a quantization-focused adapter using the low-precision runtime helpers."),
    "B012": (["gelu_fp32"], "Add a GELU backward companion kernel and keep it paired with the existing GELU path."),
    "B019": (["gather_fp32", "moe_sort_fp32", "moe_topk_fp32"], "Build the Lightning Indexer benchmark from existing gather/sort/top-k style primitives."),
    "B022": (["moe_topk_fp32", "moe_gate_route_fp16"], "Wrap the existing MoE top-k and routing logic into an init-routing benchmark adapter."),
    "B028": (["argmax_fp32"], "Use the existing argmax path as the index-selection core for maxpool-with-argmax."),
    "B029": (["argmax_fp32"], "Build maxpool-grad around the argmax/index-tracking path."),
    "B034": (["transpose_large_fp32"], "Specialize the existing transpose kernel for benchmark_007."),
    "B035": (["transpose_large_fp32"], "Specialize the existing transpose kernel for benchmark_079."),
    "B036": (["transpose_large_fp32", "permute_nhwc_nchw_fp32"], "Compose a dedicated transpose5d adapter from the existing transpose/permute blocks."),
}

PARITY_ROW_TO_KERNEL = {
    "B014": "flash_attention_cube_fp16",
    "B016": "gqa_fp16",
    "B017": "sparse_attention_local_fp16",
    "B030": "rmsnorm_fp16",
    "B011": "gelu_fp32",
    "B012": "gelu_fp32",
    "B028": "argmax_fp32",
    "B029": "argmax_fp32",
    "B039": "gather_fp32",
    "B040": "gather_fp32",
    "B041": "gather_fp32",
    "B042": "concat_fp32",
    "B043": "scatter_fp32",
    "B036": "permute_nhwc_nchw_fp32",
    "B034": "transpose_large_fp32",
    "B035": "transpose_large_fp32",
    "B044": "unsorted_segment_sum_fp32",
    "B051": "unique_i32",
}


def _column_index(cell_ref: str) -> int:
    letters = []
    for ch in cell_ref:
        if ch.isalpha():
            letters.append(ch)
        else:
            break
    acc = 0
    for ch in letters:
        acc = acc * 26 + (ord(ch.upper()) - ord("A") + 1)
    return max(acc - 1, 0)


def _shared_strings(zf: zipfile.ZipFile) -> list[str]:
    shared = "xl/sharedStrings.xml"
    if shared not in zf.namelist():
        return []
    root = ET.fromstring(zf.read(shared))
    values: list[str] = []
    for si in root.findall(f"{{{XML_NS}}}si"):
        values.append("".join(t.text or "" for t in si.iter(f"{{{XML_NS}}}t")))
    return values


def _cell_value(cell: ET.Element, shared: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        isel = cell.find(f"{{{XML_NS}}}is")
        if isel is None:
            return ""
        return "".join(t.text or "" for t in isel.iter(f"{{{XML_NS}}}t"))

    value = cell.find(f"{{{XML_NS}}}v")
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        return shared[int(value.text)]
    return value.text


def _sheet_targets(zf: zipfile.ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: rel.attrib["Target"].lstrip("/") for rel in rels}

    targets: dict[str, str] = {}
    sheets = workbook.find(f"{{{XML_NS}}}sheets")
    if sheets is None:
        return targets
    for sheet in sheets:
        rid = sheet.attrib[f"{{{REL_NS}}}id"]
        targets[sheet.attrib["name"]] = rel_map[rid]
    return targets


def load_workbook_tables(workbook_path: Path) -> dict[str, list[dict[str, str]]]:
    with zipfile.ZipFile(workbook_path) as zf:
        shared = _shared_strings(zf)
        targets = _sheet_targets(zf)
        tables: dict[str, list[dict[str, str]]] = {}
        for sheet_name in ("Summary", "Benchmark_List", "Review_Items"):
            target = targets.get(sheet_name)
            if target is None:
                continue
            root = ET.fromstring(zf.read(target))
            sheet_data = root.find(f"{{{XML_NS}}}sheetData")
            if sheet_data is None:
                tables[sheet_name] = []
                continue

            rows: list[list[str]] = []
            max_cols = 0
            for row in sheet_data.findall(f"{{{XML_NS}}}row"):
                values: list[str] = []
                for cell in row.findall(f"{{{XML_NS}}}c"):
                    idx = _column_index(cell.attrib.get("r", "A1"))
                    while len(values) <= idx:
                        values.append("")
                    values[idx] = _cell_value(cell, shared)
                max_cols = max(max_cols, len(values))
                rows.append(values)

            normalized_rows = [r + [""] * (max_cols - len(r)) for r in rows]
            if not normalized_rows:
                tables[sheet_name] = []
                continue

            header = normalized_rows[0]
            entries: list[dict[str, str]] = []
            for row in normalized_rows[1:]:
                if not any(v.strip() for v in row):
                    continue
                entry = {header[i]: row[i] for i in range(min(len(header), len(row))) if header[i]}
                entries.append(entry)
            tables[sheet_name] = entries
        return tables


def local_kernel_names() -> set[str]:
    return {path.stem for path in KERNEL_DIR.glob("*.cpp") if path.name != "README.md"}


def _impl_row(row: dict[str, str]) -> tuple[str, list[str], str]:
    bid = row["ID"]
    status = row["状态"]
    if bid in REUSE_EXISTING:
        kernels, note = REUSE_EXISTING[bid]
        return "reuse_existing", kernels, note
    if bid in SPECIALIZE_EXISTING:
        kernels, note = SPECIALIZE_EXISTING[bid]
        return "specialize_existing", kernels, note
    if status == "need_verify":
        note = row["需复核项"] or "Blocked until the workbook ambiguity is resolved."
        return "blocked_review", [], note
    return "new_kernel", [], "Requires a new kernel or wrapper implementation."


def _review_ids(review_rows: list[dict[str, str]]) -> set[str]:
    return {row["ID"] for row in review_rows if row.get("ID")}


def build_manifest(workbook_path: Path) -> dict[str, Any]:
    tables = load_workbook_tables(workbook_path)
    benchmark_rows = tables.get("Benchmark_List", [])
    review_rows = tables.get("Review_Items", [])
    review_ids = _review_ids(review_rows)
    kernels = local_kernel_names()

    benchmarks: list[dict[str, Any]] = []
    for row in benchmark_rows:
        strategy, local_kernels, impl_note = _impl_row(row)
        parity_kernel = PARITY_ROW_TO_KERNEL.get(row["ID"])
        candidate_kernel = parity_kernel or (local_kernels[0] if local_kernels else None)
        kernel_style = "mixed_tile_simt" if candidate_kernel in {
            "flash_attention_cube_fp16",
            "gqa_fp16",
            "sparse_attention_local_fp16",
            "rmsnorm_fp16",
        } else ("blocked" if strategy == "blocked_review" else "legacy")
        benchmark = {
            "id": row["ID"],
            "category": row["Benchmark分类"],
            "op_kind": row["算子种类"],
            "name": row["算子名"],
            "source": row["来源/上下文"],
            "input_shape": row["输入shape"],
            "output_shape": row["输出shape"],
            "input_dtype": row["输入dtype"],
            "output_dtype": row["输出dtype"],
            "runtime_params": row["runtime_para"],
            "selection_principle": row["用例选取原则"],
            "shape_principle": row["shape选取原则"],
            "stress_tags": [tag for tag in row["压力标签"].split(";") if tag],
            "status": row["状态"],
            "confidence": int(row["置信度"]) if row["置信度"] else 0,
            "deliverable_now": row["可直接交付实现"] == "是",
            "review_required": row["ID"] in review_ids,
            "review_note": row["需复核项"],
            "implementation_strategy": strategy,
            "local_kernels": local_kernels,
            "local_kernels_present": all(kernel in kernels for kernel in local_kernels),
            "implementation_note": impl_note,
            "parity_kernel": parity_kernel,
            "parity_covered": parity_kernel in PARITY_KERNEL_ORDER if parity_kernel else False,
            "kernel_style": kernel_style,
            "baseline_kernel": candidate_kernel,
            "candidate_kernel": candidate_kernel,
            "xdim": 1 if candidate_kernel == "rmsnorm_fp16" else (2 if candidate_kernel in {"flash_attention_cube_fp16", "gqa_fp16"} else None),
            "ydim": 1 if candidate_kernel in {"sparse_attention_local_fp16", "rmsnorm_fp16"} else (2 if candidate_kernel in {"flash_attention_cube_fp16", "gqa_fp16"} else None),
            "tile_shape": "16x16" if candidate_kernel in {"flash_attention_cube_fp16", "gqa_fp16", "sparse_attention_local_fp16"} else ("1x16" if candidate_kernel == "rmsnorm_fp16" else ""),
            "block_grouping_policy": "mixed_tile_simt" if kernel_style == "mixed_tile_simt" else "",
            "benchmark_shape": row["输入shape"],
            "smoke_shape": row["输入shape"],
            "perf_metric": "latency_us",
            "pass_threshold": "no_regression",
        }
        benchmarks.append(benchmark)

    strategy_counts = Counter(item["implementation_strategy"] for item in benchmarks)
    status_counts = Counter(item["status"] for item in benchmarks)
    op_kind_counts = Counter(item["op_kind"] for item in benchmarks)

    return {
        "workbook_path": str(workbook_path),
        "kernel_dir": str(KERNEL_DIR),
        "summary": {
            "benchmark_count": len(benchmarks),
            "deliverable_now_count": sum(1 for item in benchmarks if item["deliverable_now"]),
            "review_required_count": sum(1 for item in benchmarks if item["review_required"]),
            "parity_covered_count": sum(1 for item in benchmarks if item["parity_covered"]),
            "status_counts": dict(status_counts),
            "op_kind_counts": dict(op_kind_counts),
            "implementation_strategy_counts": dict(strategy_counts),
        },
        "benchmarks": benchmarks,
    }


def load_default_manifest() -> dict[str, Any] | None:
    if not DEFAULT_WORKBOOK.exists():
        return None
    return build_manifest(DEFAULT_WORKBOOK)


def parity_kernel_names(manifest: dict[str, Any] | None = None) -> list[str]:
    if manifest is None:
        return list(PARITY_KERNEL_ORDER)
    covered = {item["parity_kernel"] for item in manifest["benchmarks"] if item.get("parity_covered")}
    return [kernel for kernel in PARITY_KERNEL_ORDER if kernel in covered or kernel not in PARITY_ROW_TO_KERNEL.values()]


def benchmarks_by_parity_kernel(manifest: dict[str, Any] | None) -> dict[str, list[dict[str, str]]]:
    if manifest is None:
        return {}
    mapping: dict[str, list[dict[str, str]]] = {}
    for item in manifest["benchmarks"]:
        kernel = item.get("parity_kernel")
        if not kernel:
            continue
        mapping.setdefault(kernel, []).append(
            {
                "id": item["id"],
                "name": item["name"],
                "status": item["status"],
                "kernel_style": item["kernel_style"],
                "baseline_kernel": item["baseline_kernel"] or "",
                "candidate_kernel": item["candidate_kernel"] or "",
                "xdim": item["xdim"] or "",
                "ydim": item["ydim"] or "",
                "perf_metric": item["perf_metric"],
                "pass_threshold": item["pass_threshold"],
            }
        )
    return mapping


def _print_summary(manifest: dict[str, Any]) -> int:
    summary = manifest["summary"]
    print(f"Workbook: {manifest['workbook_path']}")
    print(f"Benchmarks: {summary['benchmark_count']}")
    print(f"Deliverable now: {summary['deliverable_now_count']}")
    print(f"Review required: {summary['review_required_count']}")
    print(f"Parity covered: {summary['parity_covered_count']}")
    print("Implementation strategies:")
    for key, value in sorted(summary["implementation_strategy_counts"].items()):
        print(f"  {key}: {value}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Parse the benchmark master workbook into a normalized manifest.")
    parser.add_argument("--workbook", default=str(DEFAULT_WORKBOOK), help="Path to benchmark_master_list_completed.xlsx")
    parser.add_argument("--format", choices=("summary", "json"), default="summary")
    args = parser.parse_args(argv)

    workbook = Path(args.workbook).expanduser().resolve()
    if not workbook.exists():
        raise SystemExit(f"error: workbook not found: {workbook}")

    manifest = build_manifest(workbook)
    if args.format == "json":
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    return _print_summary(manifest)


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
