#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LINXISA_ROOT="${LINXISA_ROOT:-}"
if [[ -z "$LINXISA_ROOT" ]]; then
  CAND="$(cd "$PTO_ROOT/../.." && pwd)"
  if [[ -f "$CAND/avs/qemu/run_tests.py" && -d "$CAND/workloads/generated" ]]; then
    LINXISA_ROOT="$CAND"
  fi
fi

if [[ -n "$LINXISA_ROOT" && -d "$LINXISA_ROOT/workloads/pto_kernels/kernels" ]]; then
  KERNEL_ROOT="$LINXISA_ROOT/workloads/pto_kernels/kernels"
  DEFAULT_OUT_DIR="$LINXISA_ROOT/workloads/generated/pto_asm"
else
  KERNEL_ROOT="$PTO_ROOT/kernels"
  DEFAULT_OUT_DIR="$PTO_ROOT/generated/pto_asm"
fi

OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/*.s

CXX="${CLANGXX:-${CLANG:-}}"
if [[ -z "$CXX" ]]; then
  if [[ -n "${LINXISA_ROOT:-}" && -x "$LINXISA_ROOT/compiler/llvm/build-linxisa-clang/bin/clang++" ]]; then
    CXX="$LINXISA_ROOT/compiler/llvm/build-linxisa-clang/bin/clang++"
  elif [[ -x "$HOME/llvm-project/build-linxisa-clang/bin/clang++" ]]; then
    CXX="$HOME/llvm-project/build-linxisa-clang/bin/clang++"
  fi
fi
if [[ -z "$CXX" || ! -x "$CXX" ]]; then
  echo "error: clang++ not found; set CLANG or CLANGXX" >&2
  exit 1
fi

COMMON_FLAGS=(
  -target linx64-linx-none-elf
  -O2
  -S
  -ffreestanding
  -fno-builtin
  -fno-stack-protector
  -fno-exceptions
  -fno-rtti
  -nostdlib
  -I"$PTO_ROOT/include"
)

compile_one() {
  local src="$1"
  local out="$2"
  "$CXX" "${COMMON_FLAGS[@]}" "$src" -o "$out"
}

has_tile_range() {
  local asm="$1"
  local lo="$2"
  local hi="$3"
  awk -v lo="$lo" -v hi="$hi" '
    {
      line = $0
      while (match(line, /tile[0-9]+/)) {
        tile = substr(line, RSTART + 4, RLENGTH - 4) + 0
        if (tile >= lo && tile <= hi) {
          found = 1
          exit 0
        }
        line = substr(line, RSTART + RLENGTH)
      }
    }
    END { exit(found ? 0 : 1) }
  ' "$asm"
}

has_tile_hand() {
  local asm="$1"
  local hand="$2"
  grep -Eiq "\\b${hand}#?[0-7]\\b" "$asm"
}

check_no_forbidden_tokens() {
  local asm="$1"
  local forbidden_re='((^|[^A-Za-z0-9_])L\.|set_flag|wait_flag|TSync|B\.SET|B\.WAIT)'
  if grep -Eiq "$forbidden_re" "$asm"; then
    echo "error: forbidden v0.3 or non-auto-mode token found in $asm" >&2
    exit 1
  fi
}

check_tma_descriptor_headers() {
  local asm="$1"
  awk '
    /BSTART\.T(LOAD|STORE)|BSTART\.(TMA|PAR)[[:space:]]+T(LOAD|STORE)/ {
      inblk = 1
      seen_arg = 0
      seen_ior = 0
      seen_iot = 0
      next
    }
    inblk && /BSTART\./ {
      if (!seen_arg || !seen_ior || !seen_iot) {
        exit 1
      }
      inblk = 0
    }
    inblk {
      if ($0 ~ /B\.ARG/) seen_arg = 1
      if ($0 ~ /B\.IOR/) seen_ior = 1
      if ($0 ~ /B\.IOT/) seen_iot = 1
    }
    END {
      if (inblk && (!seen_arg || !seen_ior || !seen_iot)) {
        exit 1
      }
    }
  ' "$asm" || {
    echo "error: missing B.ARG/B.IOR/B.IOT descriptor in TMA block of $asm" >&2
    exit 1
  }
}

KERNELS=(
  tload_store
  mamulb
  tmatmul_acc
  gemm
  gemm_basic
  gemm_demo
  gemm_performance
  add_custom
  flash_attention
  flash_attention_demo
  flash_attention_masked
  fa_performance
  mla_attention_demo
  flash_attention_cube_fp16
  flash_attention_cube_fp8_e4m3
  flash_attention_cube_fp4_e2m1
  flash_attention_vec_fp32
  flash_attention_vec_fp16
  mha_fp16
  gqa_fp16
  rope_apply_fp16
  attention_dropout_fp16
  flash_mla_deepseekv3_fp16
  flash_mla_deepseekv3_fp8_e4m3
  ifa_mla_seq1_fp16
  ifa_gqa_seq1_fp16
  paged_attention_mha_fp16
  paged_attention_gqa_fp16
  moe_sort_fp32
  moe_topk_fp32
  moe_gate_route_fp16
  moe_mlp_fp16
  gemm_reuse_a_fp16
  gemm_reuse_b_fp16
  gemm_reuse_ab_fp16
  flash_attention_backward_fp16
  flash_attention_backward_fp32
  sparse_attention_local_fp16
  sparse_attention_block_fp16
  rmsnorm_fp16
  relu_fp32
  sigmoid_fp32
  softmax_fp32
  tanh_fp32
  silu_fp32
  gelu_fp32
  swiglu_fp16
  layernorm_fp16
  batchnorm_fp16
  argmax_fp32
  matmul_a8w8
  matmul_a8w4
  matmul_a16w8
  matmul_a16w4
  decode_greedy_search_fp32
  decode_beam_search_fp32
  decode_temperature_scaling_fp32
  decode_topk_filter_fp32
  decode_topp_nucleus_filter_fp32
  transpose_large_fp32
  permute_nhwc_nchw_fp32
  slice_fp32
  gather_fp32
  scatter_fp32
  where_fp32
  concat_fp32
  split_fp32
  stack_fp32
  reshape_fp32
  flatten_fp32
  squeeze_fp32
  unsqueeze_fp32
  hash_table_lookup_fp32
  hash_table_insert_fp32
  unsorted_segment_sum_fp32
  unique_i32
)

for kernel in "${KERNELS[@]}"; do
  compile_one "$KERNEL_ROOT/${kernel}.cpp" "$OUT_DIR/${kernel}.s"
done

for kernel in "${KERNELS[@]}"; do
  asm="$OUT_DIR/${kernel}.s"
  check_no_forbidden_tokens "$asm"
  check_tma_descriptor_headers "$asm"

done

grep -qE "BSTART\\.TLOAD|BSTART\\.(TMA|PAR)[[:space:]]+TLOAD" "$OUT_DIR/tload_store.s"
grep -qE "BSTART\\.TSTORE|BSTART\\.(TMA|PAR)[[:space:]]+TSTORE" "$OUT_DIR/tload_store.s"
grep -qE "BSTART\\.TMATMUL|BSTART\\.(CUBE|PAR)[[:space:]]+MAMULB," "$OUT_DIR/mamulb.s"
grep -qE "BSTART\\.ACCCVT|BSTART\\.(CUBE|PAR)[[:space:]]+ACCCVT," "$OUT_DIR/mamulb.s"
grep -qE "BSTART\\.TMATMUL\\.ACC|BSTART\\.(CUBE|PAR)[[:space:]]+MAMULB\\.ACC," "$OUT_DIR/tmatmul_acc.s"
grep -qE "BSTART\\.ACCCVT|BSTART\\.(CUBE|PAR)[[:space:]]+ACCCVT," "$OUT_DIR/tmatmul_acc.s"
grep -qE "BSTART\\.TMATMUL|BSTART\\.(CUBE|PAR)[[:space:]]+MAMULB," "$OUT_DIR/gemm.s"
grep -qE "BSTART\\.TMATMUL|BSTART\\.(CUBE|PAR)[[:space:]]+MAMULB," "$OUT_DIR/flash_attention.s"
grep -qE "BSTART\\.TEPL|BSTART\\.TEXPANDS|BSTART\\.TCOLEXPAND" "$OUT_DIR/flash_attention_masked.s"

if [[ "${RUN_QEMU_TILE:-0}" == "1" ]]; then
  if [[ -z "${LINXISA_ROOT:-}" || ! -f "$LINXISA_ROOT/avs/qemu/run_tests.py" ]]; then
    echo "error: RUN_QEMU_TILE=1 requires LINXISA_ROOT with avs/qemu/run_tests.py" >&2
    exit 1
  fi
  CLANG_C="${QEMU_CLANG:-$(cd "$(dirname "$CXX")" && pwd)/clang}"
  if [[ ! -x "$CLANG_C" ]]; then
    CLANG_C="$CXX"
  fi
  QEMU_BIN="${QEMU:-}"
  if [[ -z "$QEMU_BIN" ]]; then
    for cand in "$HOME/qemu/build-tci/qemu-system-linx64" \
                "$HOME/qemu/build-linx/qemu-system-linx64"; do
      if [[ -x "$cand" ]]; then
        QEMU_BIN="$cand"
        break
      fi
    done
  fi
  QEMU_ARGS=()
  if [[ -n "$QEMU_BIN" && -x "$QEMU_BIN" ]]; then
    QEMU_ARGS+=(--qemu "$QEMU_BIN")
  fi
  CLANG="$CLANG_C" CLANGXX="$CXX" python3 "$LINXISA_ROOT/avs/qemu/run_tests.py" \
    --suite tile --timeout "${QEMU_TIMEOUT:-60}" \
    "${QEMU_ARGS[@]}" \
    --require-test-id 0x000A0001 \
    --require-test-id 0x000A0002 \
    --require-test-id 0x000A0003 \
    --require-test-id 0x000A0004 \
    --require-test-id 0x000A0005 \
    --require-test-id 0x000A0006 \
    --require-test-id 0x000A0007 \
    --require-test-id 0x000A0008 \
    --require-test-id 0x000A0009 \
    --require-test-id 0x000A000A
fi

if [[ "${RUN_PTO_PARITY:-0}" == "1" ]]; then
  LINXISA_ROOT="$LINXISA_ROOT" python3 "$PTO_ROOT/tools/run_pto_kernel_parity.py" \
    --timeout "${PTO_PARITY_TIMEOUT:-180}"
fi

echo "ok: generated PTO->Linx v0.3 assembly in $OUT_DIR"
