#!/usr/bin/env python3
"""Summarise two Nsight Compute CSV files (unfused, fused) of one TGV stage as markdown.

    ncu_summary.py ncu_unfused.csv ncu_fused.csv

Metrics come from cluster/job_ncu.sbatch: time, registers per thread, achieved
occupancy, DRAM utilisation / bytes, L2 hit rate per launch.
"""
import collections
import csv
import sys


def load(fn):
    with open(fn) as f:
        lines = [l for l in f if l.startswith('"')]          # drop ==PROF== chatter
    kernels = collections.OrderedDict()
    for r in csv.DictReader(lines):
        d = kernels.setdefault((r["ID"], r["Kernel Name"]), {})
        v = r["Metric Value"].replace(",", "")
        d[r["Metric Name"]] = float(v)
    return kernels


def short(name):
    n = name.replace("ops_par_loop_", "").replace("opensbliblock00", "")
    return n[:-len("_kernel")] if n.endswith("_kernel") else n


def table(title, k):
    out = [f"**{title}**\n",
           "| kernel | µs | regs/thread | achieved occupancy | DRAM util | DRAM MB | L2 hit |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    T = B = 0.0
    wreg = wocc = wdram = wl2 = 0.0
    for (_, name), d in k.items():
        t = d["gpu__time_duration.sum"]
        b = d["dram__bytes.sum"]
        reg = d["launch__registers_per_thread"]
        occ = d["sm__warps_active.avg.pct_of_peak_sustained_active"]
        dr = d["dram__throughput.avg.pct_of_peak_sustained_elapsed"]
        l2 = d["lts__t_sector_hit_rate.pct"]
        T, B = T + t, B + b
        wreg, wocc, wdram, wl2 = wreg + t * reg, wocc + t * occ, wdram + t * dr, wl2 + t * l2
        out.append(f"| `{short(name)}` | {t/1e3:.0f} | {reg:.0f} | {occ:.0f}% | {dr:.0f}% | {b/1e6:.0f} | {l2:.0f}% |")
    out.append(f"| **stage total** | **{T/1e3:.0f}** | {wreg/T:.0f} (time-weighted) | {wocc/T:.0f}% | {wdram/T:.0f}% | **{B/1e6:.0f}** | {wl2/T:.0f}% |")
    return "\n".join(out) + "\n", T, B


def main():
    unf, fus = load(sys.argv[1]), load(sys.argv[2])
    tu, T1, B1 = table("Unfused (17 kernels)", unf)
    tf, T2, B2 = table("Default planner (6 kernels)", fus)
    print(tu)
    print(tf)
    print(f"Stage time {T1/1e3:.0f} → {T2/1e3:.0f} µs ({T1/T2:.2f}×); DRAM traffic {B1/1e6:.0f} → {B2/1e6:.0f} MB "
          f"({B1/B2:.2f}×).\n")


if __name__ == "__main__":
    main()
