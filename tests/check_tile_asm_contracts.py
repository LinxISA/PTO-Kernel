#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from pathlib import Path

REQUIRED_TILE_KERNELS = {
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
    "flash_attention_cube_fp8_e4m3",
    "flash_attention_cube_fp4_e2m1",
    "flash_attention_vec_fp32",
    "flash_attention_vec_fp16",
    "mha_fp16",
    "gqa_fp16",
    "rope_apply_fp16",
    "attention_dropout_fp16",
    "flash_mla_deepseekv3_fp16",
    "flash_mla_deepseekv3_fp8_e4m3",
    "ifa_mla_seq1_fp16",
    "ifa_gqa_seq1_fp16",
    "paged_attention_mha_fp16",
    "paged_attention_gqa_fp16",
    "moe_sort_fp32",
    "moe_topk_fp32",
    "moe_gate_route_fp16",
    "moe_mlp_fp16",
    "gemm_reuse_a_fp16",
    "gemm_reuse_b_fp16",
    "gemm_reuse_ab_fp16",
    "flash_attention_backward_fp16",
    "flash_attention_backward_fp32",
    "sparse_attention_local_fp16",
    "sparse_attention_block_fp16",
    "rmsnorm_fp16",
}

REQUIRED_PATTERNS = [
    re.compile(r"\bBSTART\.T(LOAD|STORE|MATMUL|MATMUL\.ACC)\b"),
]
REQUIRED_VBLOCK_PATTERNS = [
    re.compile(r"\bBSTART\.(MSEQ|MPAR|VSEQ|VPAR)\b"),
    re.compile(r"\bB\.TEXT\b"),
    re.compile(r"\bB\.DIM\b"),
]
FORBIDDEN_PATTERNS = [
    re.compile(r"(^|[^A-Za-z0-9_])L\."),
    re.compile(r"\b(set_flag|wait_flag|B\.SET|B\.WAIT)\b"),
]


def check_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    errors: list[str] = []
    has_tile = any(p.search(text) for p in REQUIRED_PATTERNS)
    has_vblock = all(p.search(text) for p in REQUIRED_VBLOCK_PATTERNS)
    if path.stem in REQUIRED_TILE_KERNELS and not (has_tile or has_vblock):
        errors.append(
            f"missing required Linx block marker (tile BSTART.T* or launched "
            f"BSTART.(MSEQ|MPAR|VSEQ|VPAR) with B.TEXT/B.DIM): {path}"
        )
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
