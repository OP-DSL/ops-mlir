#!/usr/bin/env python3
"""Fusion evaluation on CloverLeaf (2D or 3D): runs the same deck under different
fusion settings and backends and tabulates launches and steady-state time.

    eval_fusion.py --exe build/apps/c/cloverleaf_2d/cloverleaf_2d \
        --deck $OPS/apps/c/CloverLeaf/clover_bm_short.in --backends seq,openmp,cuda \
        --variants off,consecutive,dag,max1,max2,max4,max8,max16,max32,ratio0.5 \
        --threads 16 --out fusion.json [--stock-exe .../cloverleaf_2d_stock] [--launch-logs DIR]

Variants: off, consecutive, dag, dag-latest; maxN = DAG planner with at most N loops per
kernel; ratioR = DAG planner with bounding-box ratio R.

Steady-state time is  execute + host-fallback  seconds from OPS_MLIR_STATS: it excludes
compilation (xDSL, kernel translation, LLVM), which is a one-off cost per distinct queue,
and the per-loop enqueue overhead, which fusion does not change. Every run must also
reproduce the stock QA value, otherwise the row is marked FAILED.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

FIXED = {
    "off":         {"OPS_MLIR_FUSION": "0"},
    "consecutive": {"OPS_MLIR_FUSION_REORDER": "0"},
    "dag":         {},
    "dag-latest":  {"OPS_MLIR_FUSION_PLACEMENT": "latest"},
    "noguard":     {"OPS_MLIR_FUSION_GUARDED": "0"},   # only loops with identical ranges fuse
}


def variant_env(name: str) -> dict:
    if name in FIXED:
        return FIXED[name]
    m = re.fullmatch(r"(?:dag-)?max(\d+)", name)
    if m:
        return {"OPS_MLIR_FUSION_MAX": m[1]}
    m = re.fullmatch(r"ratio([0-9.]+)", name)
    if m:
        return {"OPS_MLIR_FUSION_BOX_RATIO": m[1]}
    raise SystemExit(f"unknown variant {name}")


def parse(err: str, out: str) -> dict:
    r = {}
    m = re.search(r"is within ([0-9.E+-]+) %", out)
    r["qa"] = float(m.group(1)) if m else None
    r["passed"] = "PASSED" in out
    m = re.search(r"stats: (\d+) loops, (\d+) kernel launches, (\d+) flushes, (\d+) module compiles; "
                  r"seconds: compile ([0-9.]+).*? execute ([0-9.]+) \(kernel ([0-9.]+)\) "
                  r"halo ([0-9.]+) enqueue ([0-9.]+) plan ([0-9.]+)", err)
    if m:
        r.update(loops=int(m[1]), launches=int(m[2]), flushes=int(m[3]), compiles=int(m[4]),
                 compile_s=float(m[5]), execute_s=float(m[6]), kernel_s=float(m[7]),
                 enqueue_s=float(m[9]))
    m = re.search(r"coverage: (\d+) loops JIT-compiled, (\d+) through", err)
    if m:
        r.update(jit_loops=int(m[1]), host_loops=int(m[2]))
    m = re.search(r"fallback kernels \(seconds\):(.*)", err)
    if m:   # which kernels stay on the stock implementation, and what they cost
        r["host_kernels"] = {k: float(v) for k, v in re.findall(r"(\S+)=([0-9.]+)", m[1])}
    m = re.search(r"fallback total seconds: ([0-9.]+)", err)
    r["host_s"] = float(m[1]) if m else 0.0
    m = re.search(r"plus ([0-9.]+) copying", err)
    r["host_copy_s"] = float(m[1]) if m else 0.0
    if "execute_s" in r:
        r["steady_s"] = r["execute_s"] + r["host_s"] + r["host_copy_s"]
    return r


def run(exe, deck, env_extra, threads, tmp):
    work = tempfile.mkdtemp(prefix="clover_fusion_", dir=tmp)
    shutil.copy(deck, os.path.join(work, "clover.in"))
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), **env_extra)
    t = time.time()
    p = subprocess.run([os.path.abspath(exe)], cwd=work, env=env, capture_output=True, text=True)
    wall = time.time() - t
    shutil.rmtree(work, ignore_errors=True)
    return p, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--stock-exe", default=None, help="also time the stock executable (1 thread)")
    ap.add_argument("--deck", required=True)
    ap.add_argument("--backends", default="seq")
    ap.add_argument("--variants", default="off,consecutive,dag,dag-latest,dag-max32")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--launch-logs", default=None, help="directory for OPS_MLIR_LAUNCH_LOG files")
    ap.add_argument("--repeat", type=int, default=1,
                    help="repetitions; the variants are interleaved (round-robin) so that drift on a shared "
                         "machine hits all of them alike")
    args = ap.parse_args()
    if args.launch_logs:
        os.makedirs(args.launch_logs, exist_ok=True)

    results = []

    def save():
        json.dump(results, open(args.out, "w"), indent=1)

    if args.stock_exe:
        p, wall = run(args.stock_exe, args.deck, {}, 1, args.tmp)
        row = dict(backend="stock", variant="stock", wall_s=wall, rc=p.returncode, **parse(p.stderr, p.stdout))
        results.append(row)
        print(f"stock   1 thread   wall={wall:.1f}s qa={row['qa']} {'ok' if row['passed'] else 'FAILED'}", flush=True)
        save()

    backends = [b for b in args.backends.split(",") if b]
    for rep in range(args.repeat):
        for backend in backends:
            for name in args.variants.split(","):
                env = dict(variant_env(name), OPS_BACKEND=backend, OPS_MLIR_STATS="1")
                if args.launch_logs and backend == "cuda" and rep == 0:
                    env["OPS_MLIR_LAUNCH_LOG"] = os.path.join(args.launch_logs, f"{backend}_{name}.tsv")
                p, wall = run(args.exe, args.deck, env, args.threads if backend == "openmp" else 1, args.tmp)
                row = dict(backend=backend, variant=name, rep=rep, wall_s=wall, rc=p.returncode,
                           **parse(p.stderr, p.stdout))
                if p.returncode != 0 or not row["passed"]:
                    row["failed"] = True
                results.append(row)
                print(f"{backend:7s} {name:12s} rep={rep} launches={row.get('launches')} "
                      f"kernel={row.get('kernel_s', float('nan')):.2f}s host={row.get('host_s', 0):.2f}s "
                      f"copy={row.get('host_copy_s', 0):.2f}s steady={row.get('steady_s', float('nan')):.2f}s "
                      f"qa={row['qa']} {'FAILED' if row.get('failed') else 'ok'}", flush=True)
                save()


if __name__ == "__main__":
    sys.exit(main())
