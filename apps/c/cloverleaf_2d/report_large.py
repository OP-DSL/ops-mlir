#!/usr/bin/env python3
"""Markdown tables and SVG charts for the larger-CloverLeaf study.

    report_large.py --eval DIR --outdir docs [--tag a100]

DIR holds what cluster/job_clover_large.sbatch and job_clover_ncu.sbatch produced:
clover_large_{stock,cuda,openmp}_{2d,3d}.json, launch_logs_{2d,3d}/cuda_<variant>.tsv and
clover_ncu_{2d,3d}.json. Whatever is missing is skipped.
"""
import argparse
import collections
import json
import os
import re
import sys

PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#72b7b2", "#9d755d", "#bab0ac"]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def variant_key(v):
    m = re.fullmatch(r"max(\d+)", v)
    if m:
        return (1, int(m[1]))
    m = re.fullmatch(r"ratio([0-9.]+)", v)
    if m:
        return (3, float(m[1]))
    return {"off": (0, 0), "consecutive": (2, 0), "noguard": (2, 1), "dag": (2, 2)}.get(v, (2, 3))


# ------------------------------------------------------------------ SVG
def svg_lines(title, xs, series, xlabel, ylabel, width=640, height=300):
    """xs are category labels (evenly spaced); series: name -> list of values (None = gap)."""
    left, right, top, bottom = 70, 150, 40, 50
    pw, ph = width - left - right, height - top - bottom
    vals = [v for s in series.values() for v in s if v is not None]
    vmax = max(vals) * 1.1 if vals else 1.0
    X = lambda i: left + pw * (i + 0.5) / len(xs)
    Y = lambda v: top + ph - ph * v / vmax
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'font-family="sans-serif" font-size="12">',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<text x="{(width - right + left) / 2}" y="22" text-anchor="middle" font-size="14" '
           f'font-weight="bold">{title}</text>']
    for i in range(5):
        v = vmax * i / 4
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{Y(v)}" y2="{Y(v)}" stroke="#ddd"/>')
        out.append(f'<text x="{left - 6}" y="{Y(v) + 4}" text-anchor="end">{v:.3g}</text>')
    for i, x in enumerate(xs):
        out.append(f'<text x="{X(i)}" y="{top + ph + 16}" text-anchor="middle">{x}</text>')
    out.append(f'<text x="{left + pw / 2}" y="{height - 8}" text-anchor="middle">{xlabel}</text>')
    out.append(f'<text x="14" y="{top + ph / 2}" transform="rotate(-90 14 {top + ph / 2})" '
               f'text-anchor="middle">{ylabel}</text>')
    for k, (name, ys) in enumerate(series.items()):
        col = PALETTE[k % len(PALETTE)]
        pts = " ".join(f"{X(i)},{Y(v)}" for i, v in enumerate(ys) if v is not None)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        for i, v in enumerate(ys):
            if v is not None:
                out.append(f'<circle cx="{X(i)}" cy="{Y(v)}" r="3" fill="{col}"/>')
        ly = top + 14 + 16 * k
        out.append(f'<rect x="{width - right + 10}" y="{ly - 9}" width="12" height="12" fill="{col}"/>')
        out.append(f'<text x="{width - right + 28}" y="{ly + 2}">{name}</text>')
    out.append('</svg>')
    return "\n".join(out)


def svg_scatter(title, groups, xlabel, ylabel, width=640, height=320, xmax=None, ymax=None):
    """groups: name -> list of (x, y)."""
    left, right, top, bottom = 70, 150, 40, 50
    pw, ph = width - left - right, height - top - bottom
    xs = [p[0] for g in groups.values() for p in g]
    ys = [p[1] for g in groups.values() for p in g]
    xmax = xmax or (max(xs) * 1.08 if xs else 1)
    ymax = ymax or (max(ys) * 1.1 if ys else 1)
    X = lambda x: left + pw * x / xmax
    Y = lambda y: top + ph - ph * y / ymax
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'font-family="sans-serif" font-size="12">',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<text x="{(width - right + left) / 2}" y="22" text-anchor="middle" font-size="14" '
           f'font-weight="bold">{title}</text>']
    for i in range(5):
        y = ymax * i / 4
        x = xmax * i / 4
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{Y(y)}" y2="{Y(y)}" stroke="#ddd"/>')
        out.append(f'<text x="{left - 6}" y="{Y(y) + 4}" text-anchor="end">{y:.3g}</text>')
        out.append(f'<text x="{X(x)}" y="{top + ph + 16}" text-anchor="middle">{x:.3g}</text>')
    out.append(f'<text x="{left + pw / 2}" y="{height - 8}" text-anchor="middle">{xlabel}</text>')
    out.append(f'<text x="14" y="{top + ph / 2}" transform="rotate(-90 14 {top + ph / 2})" '
               f'text-anchor="middle">{ylabel}</text>')
    for k, (name, pts) in enumerate(groups.items()):
        col = PALETTE[k % len(PALETTE)]
        for x, y in pts:
            out.append(f'<circle cx="{X(x)}" cy="{Y(y)}" r="3.5" fill="{col}" fill-opacity="0.65"/>')
        ly = top + 14 + 16 * k
        out.append(f'<circle cx="{width - right + 16}" cy="{ly - 3}" r="5" fill="{col}"/>')
        out.append(f'<text x="{width - right + 28}" y="{ly + 2}">{name}</text>')
    out.append('</svg>')
    return "\n".join(out)


# ------------------------------------------------------------------ tables
def timing_section(w, dim, ev, outdir, tag, imgs):
    cuda = load(f"{ev}/clover_large_cuda_{dim}.json")
    omp = load(f"{ev}/clover_large_openmp_{dim}.json")
    stock = load(f"{ev}/clover_large_stock_{dim}.json")
    if not (cuda or omp):
        return
    deck = {"2d": "2D, 3840², `clover_bm16_short.in`", "3d": "3D, 256³, `clover_bm1s_short.in`"}[dim]
    w(f"### {dim.upper()}: {deck}\n")
    if stock:
        s = stock[0]
        w(f"Stock implementation, one thread: **{s['wall_s']:.0f} s** wall, QA {s['qa']:.3e} % "
          f"({'PASSED' if s['passed'] else 'FAILED'}).\n")
    for name, rows in (("CUDA (A100)", cuda), ("OpenMP (16 threads)", omp)):
        if not rows:
            continue
        base = next((r for r in rows if r["variant"] == "off"), rows[0])
        w(f"**{name}**\n")
        w("| variant | launches | kernel s | host-fallback s | steady state s | vs off | compile s | QA |")
        w("|---|---:|---:|---:|---:|---:|---:|---|")
        for r in sorted(rows, key=lambda r: variant_key(r["variant"])):
            if r.get("failed"):
                w(f"| {r['variant']} | FAILED | | | | | | |")
                continue
            w(f"| {r['variant']} | {r['launches']} | {r['kernel_s']:.2f} | {r['host_s']:.2f} | "
              f"{r['steady_s']:.2f} | {r['steady_s'] / base['steady_s'] - 1:+.0%} | {r['compile_s']:.0f} | "
              f"{r['qa']:.3e} |")
        w("")
    if cuda:
        sweep = sorted([r for r in cuda if re.fullmatch(r"max\d+", r["variant"]) or r["variant"] == "off"],
                       key=lambda r: variant_key(r["variant"]))
        xs = [("1" if r["variant"] == "off" else r["variant"][3:]) for r in sweep]
        svg = svg_lines(f"{dim.upper()} on the A100: loops per kernel", xs,
                        {"steady state s": [r["steady_s"] for r in sweep],
                         "launches / 1000": [r["launches"] / 1000 for r in sweep]},
                        "max loops per kernel (OPS_MLIR_FUSION_MAX)", "s  |  thousands of launches")
        fn = f"img/clover_large_sweep_{dim}{tag}.svg"
        open(os.path.join(outdir, fn), "w").write(svg)
        w(f"![sweep]({fn})\n")


def launch_histogram(w, dim, ev):
    d = f"{ev}/launch_logs_{dim}"
    if not os.path.isdir(d):
        return
    rows = []
    for v in ("off", "max2", "max3", "max4", "max8", "max16", "max64"):
        p = f"{d}/cuda_{v}.tsv"
        if not os.path.exists(p):
            continue
        sizes = collections.Counter()
        tsec = collections.defaultdict(float)
        for line in open(p):
            f = line.rstrip("\n").split("\t")
            n = len(f[4].split("+"))
            sizes[n] += 1
            tsec[n] += float(f[1])
        rows.append((v, sizes, tsec))
    if not rows:
        return
    w(f"**{dim.upper()}: launches by number of loops fused** (whole run; seconds are launch + sync)\n")
    w("| variant | " + " | ".join(f"{n} loop{'s' if n > 1 else ''}" for n in range(1, 9)) + " | >8 | largest |")
    w("|---|" + "---:|" * 10)
    for v, sizes, tsec in rows:
        cells = [str(sizes.get(n, 0)) for n in range(1, 9)]
        big = sum(c for n, c in sizes.items() if n > 8)
        w(f"| {v} | " + " | ".join(cells) + f" | {big} | {max(sizes)} |")
    w("")


def ncu_section(w, dim, ev, outdir, tag):
    data = load(f"{ev}/clover_ncu_{dim}.json")
    if not data:
        return
    w(f"### {dim.upper()}: one steady-state time step under Nsight Compute\n")
    w("| variant | kernels | step time ms | DRAM MB | time-weighted regs | max regs | "
      "time-weighted occupancy | kernels ≥ 128 regs | local-memory sectors |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    summary = {}
    for r in sorted(data, key=lambda r: variant_key(r["variant"])):
        ks = r["kernels"]
        T = sum(k["time_us"] for k in ks)
        wr = sum(k["time_us"] * k["regs"] for k in ks) / T
        wo = sum(k["time_us"] * k["occupancy"] for k in ks) / T
        summary[r["variant"]] = dict(n=len(ks), T=T)
        w(f"| {r['variant']} | {len(ks)} | {T / 1e3:.2f} | {sum(k['dram_mb'] for k in ks):.0f} | {wr:.0f} | "
          f"{max(k['regs'] for k in ks):.0f} | {wo:.0f}% | {sum(1 for k in ks if k['regs'] >= 128)} | "
          f"{sum(k['local_sectors'] for k in ks):.3g} |")
    w("")
    # registers against fusion size, all variants pooled
    by_size = collections.defaultdict(list)
    for r in data:
        for k in r["kernels"]:
            by_size[k["nloops"]].append(k)
    w("Pooled over all variants, by number of loops in the kernel:\n")
    w("| loops fused | kernels | mean regs | max regs | mean occupancy | mean µs | DRAM MB / kernel |")
    w("|---:|---:|---:|---:|---:|---:|---:|")
    for n in sorted(by_size):
        ks = by_size[n]
        m = lambda key: sum(k[key] for k in ks) / len(ks)
        w(f"| {n} | {len(ks)} | {m('regs'):.0f} | {max(k['regs'] for k in ks):.0f} | {m('occupancy'):.0f}% | "
          f"{m('time_us'):.0f} | {m('dram_mb'):.1f} |")
    w("")
    groups = {}
    for r in sorted(data, key=lambda r: variant_key(r["variant"])):
        groups[r["variant"]] = [(k["nloops"], k["regs"]) for k in r["kernels"]]
    svg = svg_scatter(f"{dim.upper()}: registers per thread vs loops fused", groups,
                      "loops fused into the kernel", "registers / thread", ymax=255)
    fn = f"img/clover_regs_{dim}{tag}.svg"
    open(os.path.join(outdir, fn), "w").write(svg)
    w(f"![registers]({fn})\n")
    xs = [v for v in summary]
    svg = svg_lines(f"{dim.upper()}: one time step", [("1" if v == "off" else v[3:]) for v in xs],
                    {"step time ms": [summary[v]["T"] / 1e3 for v in xs],
                     "kernels / 10": [summary[v]["n"] / 10 for v in xs]},
                    "max loops per kernel", "ms  |  tens of kernels")
    fn = f"img/clover_step_{dim}{tag}.svg"
    open(os.path.join(outdir, fn), "w").write(svg)
    w(f"![step]({fn})\n")
    # the heaviest kernels of the default setting
    default = next((r for r in data if r["variant"] == "max8"), data[-1])
    w(f"Ten slowest kernels of the default setting (`{default['variant']}`):\n")
    w("| loops fused | µs | regs | limit by regs | occupancy | DRAM util | DRAM MB | members |")
    w("|---:|---:|---:|---:|---:|---:|---:|---|")
    for k in sorted(default["kernels"], key=lambda k: -k["time_us"])[:10]:
        mem = ", ".join(sorted(set(k["members"])))
        w(f"| {k['nloops']} | {k['time_us']:.0f} | {k['regs']:.0f} | {k['occ_limit_regs']:.0f} | "
          f"{k['occupancy']:.0f}% | {k['dram_pct']:.0f}% | {k['dram_mb']:.0f} | {mem[:90]} |")
    w("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tag = f"_{a.tag}" if a.tag else ""
    os.makedirs(os.path.join(a.outdir, "img"), exist_ok=True)
    w = lambda s="": print(s)
    w("## Timing\n")
    for dim in ("2d", "3d"):
        timing_section(w, dim, a.eval, a.outdir, tag, None)
        launch_histogram(w, dim, a.eval)
    w("## Nsight Compute\n")
    for dim in ("2d", "3d"):
        ncu_section(w, dim, a.eval, a.outdir, tag)


if __name__ == "__main__":
    sys.exit(main())
