#!/usr/bin/env python3
"""Fusion evaluation on CloverLeaf (2D or 3D): runs the same deck under different
fusion settings and backends and tabulates launches and steady-state time.

    eval_fusion.py --exe build/apps/c/cloverleaf_2d/cloverleaf_2d \
        --deck $OPS/apps/c/CloverLeaf/clover_bm_short.in --backends seq,openmp,cuda \
        --threads 16 --out fusion.json

Steady-state time is  execute + host-fallback  seconds from OPS_MLIR_STATS: it excludes
compilation (xDSL, kernel translation, LLVM), which is a one-off cost per distinct
queue, and the per-loop enqueue overhead, which fusion does not change. Every run must
also reproduce the stock QA value, otherwise the row is marked FAILED.
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

VARIANTS = {
    # name: environment
    "off":         {"OPS_MLIR_FUSION": "0"},
    "consecutive": {"OPS_MLIR_FUSION_REORDER": "0"},
    "dag":         {},
    "dag-latest":  {"OPS_MLIR_FUSION_PLACEMENT": "latest"},
    "dag-max16":   {"OPS_MLIR_FUSION_MAX": "16"},
    "dag-max32":   {"OPS_MLIR_FUSION_MAX": "32"},
    "dag-max64":   {"OPS_MLIR_FUSION_MAX": "64"},
}


def parse(err: str, out: str) -> dict:
    r = {}
    m = re.search(r"is within ([0-9.E+-]+) %", out)
    r["qa"] = float(m.group(1)) if m else None
    r["passed"] = "PASSED" in out
    m = re.search(r"stats: (\d+) loops, (\d+) kernel launches, (\d+) flushes, (\d+) module compiles; "
                  r"seconds: compile ([0-9.]+).*? execute ([0-9.]+) halo ([0-9.]+) enqueue ([0-9.]+) plan ([0-9.]+)", err)
    if m:
        r.update(loops=int(m[1]), launches=int(m[2]), flushes=int(m[3]), compiles=int(m[4]),
                 compile_s=float(m[5]), execute_s=float(m[6]), enqueue_s=float(m[8]))
    m = re.search(r"coverage: (\d+) loops JIT-compiled, (\d+) through", err)
    if m:
        r.update(jit_loops=int(m[1]), host_loops=int(m[2]))
    m = re.search(r"fallback total seconds: ([0-9.]+)", err)
    r["host_s"] = float(m[1]) if m else 0.0
    if "execute_s" in r:
        r["steady_s"] = r["execute_s"] + r["host_s"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--backends", default="seq")
    ap.add_argument("--variants", default="off,consecutive,dag,dag-latest,dag-max16,dag-max32")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args()

    results = []
    for backend in args.backends.split(","):
        for name in args.variants.split(","):
            work = tempfile.mkdtemp(prefix="clover_fusion_", dir=args.tmp)
            shutil.copy(args.deck, os.path.join(work, "clover.in"))
            env = dict(os.environ, OPS_BACKEND=backend, OMP_NUM_THREADS=str(args.threads if backend == "openmp" else 1),
                       OPS_MLIR_STATS="1", **VARIANTS[name])
            t = time.time()
            p = subprocess.run([os.path.abspath(args.exe)], cwd=work, env=env, capture_output=True, text=True)
            wall = time.time() - t
            row = dict(backend=backend, variant=name, wall_s=wall, rc=p.returncode, **parse(p.stderr, p.stdout))
            if p.returncode != 0 or not row["passed"]:
                row["failed"] = True
            results.append(row)
            print(f"{backend:7s} {name:12s} launches={row.get('launches')} steady={row.get('steady_s', float('nan')):.2f}s "
                  f"compile={row.get('compile_s', float('nan')):.1f}s qa={row['qa']} {'FAILED' if row.get('failed') else 'ok'}",
                  flush=True)
            shutil.rmtree(work, ignore_errors=True)
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
