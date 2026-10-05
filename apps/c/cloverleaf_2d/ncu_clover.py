#!/usr/bin/env python3
"""Per-kernel Nsight Compute profile of one steady-state CloverLeaf time step, for several
fusion settings, joined to the loops each kernel fuses.

    ncu_clover.py --exe build/apps/c/cloverleaf_2d/cloverleaf_2d \
        --deck $OPS/apps/c/CloverLeaf/clover_bm16_short.in --variants off,max2,max8,max32 \
        --out clover_ncu_2d.json

For every variant:
  1. a run to step 2 with OPS_MLIR_LAUNCH_LOG tells how many kernels precede step 3;
  2. a run to step 3 under `ncu --launch-skip <that>` profiles everything after it, i.e.
     step 3 (the later steps are identical; a shorter run only saves time);
  3. the k-th profiled CUDA kernel is matched to the k-th line of the run's launch log,
     which names the generated function and the loops fused into it.
The output is one JSON list per variant with registers/thread, achieved occupancy, DRAM
traffic, L2 hit rate and local-memory (spill) traffic for each kernel of the step.
"""
import argparse
import collections
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

METRICS = ",".join([
    "launch__registers_per_thread",
    "launch__occupancy_limit_registers",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum",
    "gpu__time_duration.sum",
    "lts__t_sector_hit_rate.pct",
    "launch__grid_size",
    "launch__block_size",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
])


def variant_env(name):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_fusion import variant_env as ve
    return ve(name)


def deck_with_steps(deck, steps, dest):
    text = open(deck).read()
    text = re.sub(r"end_step\s*=\s*\d+", f"end_step={steps}", text)
    text = re.sub(r"end_time\s*=\s*[0-9.eE+-]+", "end_time=1000000.0", text)
    open(dest, "w").write(text)


def read_log(path):
    rows = []
    for line in open(path):
        kind, idx, sec, nbytes, func, members = line.rstrip("\n").split("\t")
        rows.append(dict(kind=kind, idx=int(idx), seconds=float(sec), model_bytes=float(nbytes), func=func,
                         members=members.split("+")))
    return rows


def parse_ncu(stdout):
    lines = [l for l in stdout.splitlines() if l.startswith('"')]
    kernels = collections.OrderedDict()
    for r in csv.DictReader(lines):
        d = kernels.setdefault(r["ID"], {"name": r["Kernel Name"]})
        v = r["Metric Value"].replace(",", "")
        try:
            d[r["Metric Name"]] = float(v)
        except ValueError:
            d[r["Metric Name"]] = v
    return list(kernels.values())


def profile(args, variant, workdir):
    env = dict(os.environ, OPS_BACKEND="cuda", OMP_NUM_THREADS="1", **variant_env(variant))
    deck2, deck3 = (os.path.join(workdir, f"deck{n}.in") for n in (2, 3))
    deck_with_steps(args.deck, 2, deck2)
    deck_with_steps(args.deck, 3, deck3)

    def sandbox(deck):
        d = tempfile.mkdtemp(prefix="ncu_", dir=args.tmp)
        shutil.copy(deck, os.path.join(d, "clover.in"))
        return d

    # 1. kernels before step 3
    d = sandbox(deck2)
    log2 = os.path.join(workdir, f"{variant}_2.tsv")
    subprocess.run([os.path.abspath(args.exe)], cwd=d, env=dict(env, OPS_MLIR_LAUNCH_LOG=log2),
                   capture_output=True, text=True, check=True)
    shutil.rmtree(d, ignore_errors=True)
    skip = sum(1 for r in read_log(log2) if r["kind"] == "G")

    # 2. profile what follows
    d = sandbox(deck3)
    log3 = os.path.join(workdir, f"{variant}_3.tsv")
    p = subprocess.run(["ncu", "--csv", "--target-processes", "application-only", "--metrics", METRICS, "--launch-skip", str(skip), args.exe_abs],
                       cwd=d, env=dict(env, OPS_MLIR_LAUNCH_LOG=log3), capture_output=True, text=True)
    shutil.rmtree(d, ignore_errors=True)
    if p.returncode != 0:
        raise SystemExit(f"ncu failed for {variant}:\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}")
    profiled = parse_ncu(p.stdout)
    launches = [r for r in read_log(log3) if r["kind"] == "G"][skip:]

    # 3. join
    if len(profiled) != len(launches):
        print(f"  warning: {len(profiled)} profiled kernels vs {len(launches)} logged launches "
              f"({variant}); matching by order up to the shorter", file=sys.stderr)
    kernels = []
    mismatches = 0
    for k, l in zip(profiled, launches):
        if l["func"] not in k["name"]:
            mismatches += 1
        kernels.append(dict(
            func=l["func"], members=l["members"], nloops=len(l["members"]),
            time_us=k["gpu__time_duration.sum"] / 1e3,       # ncu reports ns
            regs=k["launch__registers_per_thread"],
            occ_limit_regs=k["launch__occupancy_limit_registers"],
            occupancy=k["sm__warps_active.avg.pct_of_peak_sustained_active"],
            dram_pct=k["dram__throughput.avg.pct_of_peak_sustained_elapsed"],
            dram_mb=k["dram__bytes.sum"] / 1e6,
            l2_hit=k["lts__t_sector_hit_rate.pct"],
            grid=k["launch__grid_size"], block=k["launch__block_size"],
            local_sectors=k["l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"]
                          + k["l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"],
            model_mb=l["model_bytes"] / 1e6,
        ))
    return dict(variant=variant, skipped=skip, profiled=len(profiled), logged=len(launches),
                name_mismatches=mismatches, kernels=kernels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--variants", default="off,max2,max8,max32")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args()
    args.exe_abs = os.path.abspath(args.exe)
    workdir = tempfile.mkdtemp(prefix="ncu_logs_", dir=args.tmp)
    results = []
    for v in args.variants.split(","):
        print(f"profiling {v} ...", flush=True)
        r = profile(args, v, workdir)
        tot = sum(k["time_us"] for k in r["kernels"])
        print(f"  {len(r['kernels'])} kernels, {tot/1e3:.2f} ms, "
              f"max regs {max(k['regs'] for k in r['kernels']):.0f}, "
              f"name mismatches {r['name_mismatches']}", flush=True)
        results.append(r)
        json.dump(results, open(args.out, "w"), indent=1)
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
