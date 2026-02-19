#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PTO_ROOT = SCRIPT.parents[1]

KERNEL_NAMES = [
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
]

DIGEST_RE = re.compile(r"PTO_DIGEST\s+([A-Za-z0-9_]+)\s+0x([0-9A-Fa-f]+)")


def is_linxisa_root(path: Path) -> bool:
    return (path / "avs" / "qemu" / "run_tests.py").exists() and (
        path / "workloads" / "generated"
    ).exists()


def detect_linxisa_root() -> Path | None:
    env = os.environ.get("LINXISA_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if is_linxisa_root(p):
            return p
        raise SystemExit(f"error: LINXISA_ROOT is not a valid linx-isa root: {p}")

    for anc in PTO_ROOT.parents:
        if is_linxisa_root(anc):
            return anc
    return None


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def parse_digests(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in DIGEST_RE.finditer(text):
        out[m.group(1)] = "0x" + m.group(2).upper()
    return out


def host_compiler_works(compiler: str) -> bool:
    p = subprocess.run(
        [compiler, "-x", "c++", "-std=c++17", "-", "-c", "-o", os.devnull],
        input="int main(){return 0;}\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return p.returncode == 0


def pick_clangxx(linxisa_root: Path | None) -> str:
    env = os.environ.get("CLANGXX")
    if env:
        if host_compiler_works(env):
            return env
        raise SystemExit(
            f"error: CLANGXX={env} cannot compile host C++ code; "
            "set CLANGXX to a host compiler (for example /usr/bin/clang++)"
        )

    candidates: list[str] = []
    clangxx = shutil.which("clang++")
    if clangxx:
        candidates.append(clangxx)
    cxx = shutil.which("c++")
    if cxx:
        candidates.append(cxx)

    if linxisa_root:
        local = linxisa_root / "compiler" / "llvm" / "build-linxisa-clang" / "bin" / "clang++"
        if local.exists():
            candidates.append(str(local))

    for cand in candidates:
        if host_compiler_works(cand):
            return cand

    raise SystemExit(
        "error: no usable host C++ compiler found; "
        "install clang++ or set CLANGXX=/path/to/host-clang++"
    )


def kernel_sources(linxisa_root: Path | None) -> list[Path]:
    if linxisa_root:
        base = linxisa_root / "workloads" / "pto_kernels" / "kernels"
    else:
        base = PTO_ROOT / "kernels"
    return [base / f"{name}.cpp" for name in KERNEL_NAMES]


def build_and_run_host(clangxx: str, host_bin: Path, linxisa_root: Path) -> tuple[dict[str, str], str]:
    harness = linxisa_root / "avs" / "qemu" / "tests" / "16_pto_kernel_parity.cpp"
    include_dir = linxisa_root / "workloads" / "pto_kernels" / "include"

    if not harness.exists():
        raise SystemExit(f"error: missing parity harness: {harness}")

    sources = [str(harness), *[str(p) for p in kernel_sources(linxisa_root)]]
    cmd = [
        clangxx,
        "-std=c++17",
        "-O2",
        "-DPTO_HOST_SIM=1",
        "-DPTO_QEMU_SMOKE=1",
        f"-I{include_dir}",
        *sources,
        "-o",
        str(host_bin),
    ]
    p = run(cmd)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit("error: host parity build failed")

    r = run([str(host_bin)])
    if r.returncode != 0:
        sys.stderr.write(r.stdout)
        sys.stderr.write(r.stderr)
        raise SystemExit("error: host parity binary failed")

    text = (r.stdout or "") + "\n" + (r.stderr or "")
    return parse_digests(text), text


def run_qemu_suite(linxisa_root: Path, timeout_s: float) -> tuple[dict[str, str], str, list[str]]:
    cmd = [
        "python3",
        str(linxisa_root / "avs" / "qemu" / "run_tests.py"),
        "--suite",
        "pto_parity",
        "--timeout",
        str(timeout_s),
        "--verbose",
    ]
    compile_and_run_timeout = timeout_s + 120.0
    try:
        p = run(cmd, cwd=linxisa_root, timeout=compile_and_run_timeout)
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + "\n" + (exc.stderr or "")
        if out:
            sys.stderr.write(out)
        raise SystemExit(
            "error: qemu pto_parity suite timed out "
            f"(timeout={compile_and_run_timeout:.1f}s)"
        )
    text = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        sys.stderr.write(text)
        raise SystemExit(f"error: qemu pto_parity suite failed (exit={p.returncode})")
    return parse_digests(text), text, cmd


def write_reports(host: dict[str, str], qemu: dict[str, str], qemu_cmd: list[str], host_log: str, qemu_log: str, out_dir: Path) -> tuple[Path, Path, bool]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "pto_kernel_parity_latest.json"
    md_path = out_dir / "pto_kernel_parity_latest.md"

    rows = []
    ok = True
    for name in KERNEL_NAMES:
        hv = host.get(name)
        qv = qemu.get(name)
        match = hv is not None and qv is not None and hv == qv
        if not match:
            ok = False
        rows.append(
            {
                "kernel": name,
                "host_digest": hv,
                "qemu_digest": qv,
                "match": match,
            }
        )

    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "profile": "smoke",
        "expected_kernels": KERNEL_NAMES,
        "host_digest_count": len(host),
        "qemu_digest_count": len(qemu),
        "all_match": ok,
        "qemu_command": qemu_cmd,
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# PTO Kernel Parity (Host vs QEMU)",
        "",
        f"- Generated (UTC): `{payload['generated_at_utc']}`",
        "- Profile: `PTO_QEMU_SMOKE=1`",
        f"- All match: `{'YES' if ok else 'NO'}`",
        "",
        "| Kernel | Host Digest | QEMU Digest | Match |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['kernel']}` | `{r['host_digest'] or 'MISSING'}` | `{r['qemu_digest'] or 'MISSING'}` | `{'yes' if r['match'] else 'no'}` |"
        )
    lines += [
        "",
        "## Raw Logs",
        "",
        "### Host",
        "```text",
        host_log.strip(),
        "```",
        "",
        "### QEMU",
        "```text",
        qemu_log.strip(),
        "```",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return json_path, md_path, ok


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run PTO kernel parity (host-sim vs QEMU).")
    parser.add_argument("--timeout", type=float, default=180.0, help="QEMU timeout seconds")
    parser.add_argument("--out-dir", default=None, help="Optional report output directory")
    args = parser.parse_args(argv)

    linxisa_root = detect_linxisa_root()
    if linxisa_root is None:
        raise SystemExit(
            "error: this parity runner requires a linx-isa integration root. "
            "Set LINXISA_ROOT when running outside the superproject."
        )

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else linxisa_root / "workloads" / "generated"

    clangxx = pick_clangxx(linxisa_root)
    host_bin = out_dir / "pto_kernel_parity_host"

    host_digests, host_log = build_and_run_host(clangxx, host_bin, linxisa_root)
    qemu_digests, qemu_log, qemu_cmd = run_qemu_suite(linxisa_root, args.timeout)

    json_path, md_path, ok = write_reports(
        host_digests,
        qemu_digests,
        qemu_cmd,
        host_log,
        qemu_log,
        out_dir,
    )

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if not ok:
        print("parity mismatch detected")
        return 1
    print("parity matched for all kernels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
