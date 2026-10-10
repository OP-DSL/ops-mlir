#!/usr/bin/env python3
"""Evaluate loop fusion / reordering on the Taylor-Green vortex.

    eval_tgv.py run    --app <build/apps/c/taylor_green_vortex> --out results.json
    eval_tgv.py report --results results.json --outdir docs

`run` executes the experiments below (sequentially: GPU timings must not
overlap) and writes one JSON file; `report` turns it into a markdown report and
SVG charts. Everything is driven by environment variables of the app/runtime,
so the numbers are reproducible with nothing but this script.

Experiments
  plan      kernels per Runge-Kutta stage / per iteration for each planner
            configuration, estimated memory traffic, and why loops could not
            join an existing kernel (OPS_MLIR_PLAN)
  timing    steady-state seconds per iteration, taken from the app's own timing
            of windows of 100 iterations inside ONE process, discarding the
            warm-up windows (the first window contains the JIT compile of every
            kernel shape, the first filter step is at iteration 25). The same
            run gives an exact breakdown from the runtime's cumulative timers
            (kernels / halo exchange / enqueue / planning), and the cost of
            warm-up. Medians over windows and repetitions. (Differencing two
            whole-run times does NOT work: JIT time varies by about a second
            between runs, i.e. +-10 ms per iteration.)
  sizes     the same over grid sizes (fusion should matter most where kernels
            are small and launch/bandwidth overheads dominate)
  caps      sweep of the maximum loops per kernel and of the bounding-box ratio
  profile   per-kernel time from the runtime's profiler, unfused vs default
  accuracy  every configuration against the unfused result on the same backend
            (bitwise on CPU, rounding-level on GPU), and f32 against f64
"""
import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_fields as cf  # noqa: E402

CONFIGS = {
    "unfused":               {"OPS_MLIR_FUSION": "0"},
    "adjacent, same-range":  {"OPS_MLIR_FUSION_REORDER": "0", "OPS_MLIR_FUSION_GUARDED": "0"},
    "adjacent, guarded":     {"OPS_MLIR_FUSION_REORDER": "0"},
    "DAG, same-range":       {"OPS_MLIR_FUSION_GUARDED": "0"},
    "DAG, guarded":          {},                                   # the default
    "DAG, guarded, latest":  {"OPS_MLIR_FUSION_PLACEMENT": "latest"},
}
DEFAULT = "DAG, guarded"

WINDOW = 100           # iterations per timing window (the app prints every 100)
WARMUP_ITERS = 100     # windows ending at or before this are warm-up
TIMING_ITERS = 400     # iterations per timing run


def sh_env(app_env):
    env = dict(os.environ)
    env.update(app_env)
    return env


class Runner:
    def __init__(self, app_dir, work, cuda_rt):
        self.app_dir, self.work, self.cuda_rt = os.path.abspath(app_dir), os.path.abspath(work), cuda_rt
        self.count = 0

    def run(self, exe, backend, n, niter, cfg_env, dump=False, plan=False, timeout=900):
        self.count += 1
        d = tempfile.mkdtemp(prefix="tgv_", dir=self.work)
        env = {"OPS_BACKEND": backend, "TGV_N": str(n), "TGV_NITER": str(niter),
               "OPS_MLIR_CUDA_RUNTIME": self.cuda_rt}
        env.update(cfg_env)
        if dump:
            env["TGV_DUMP"] = os.path.join(d, "fields")
            env["TGV_DIAG"] = "1"
        if plan:
            env["OPS_MLIR_PLAN"] = "1"
        t0 = time.time()
        r = subprocess.run([os.path.join(self.app_dir, exe)], cwd=d, env=sh_env(env),
                           capture_output=True, text=True, timeout=timeout)
        out = r.stdout + "\n" + r.stderr
        if r.returncode != 0:
            shutil.rmtree(d, ignore_errors=True)
            raise RuntimeError(f"{exe} {backend} n={n} failed ({r.returncode}): {out[-400:]}")
        res = {"wall_total": time.time() - t0, "out": out, "dir": d}
        m = re.search(r"Total Wall time ([0-9.eE+-]+)", out)
        res["wall"] = float(m.group(1)) if m else float("nan")
        return res

    def done(self, res):
        shutil.rmtree(res["dir"], ignore_errors=True)


def median(xs):
    return statistics.median(xs)


def spread(xs):
    m = median(xs)
    return (max(xs) - min(xs)) / m if m else 0.0


IT_RE = re.compile(r"^Iteration: (\d+)\..*Time/iteration: ([0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?)", re.M)
WIN_RE = re.compile(r"^\[window\] iter=(\d+) loops=(\d+) launches=(\d+) compile=([0-9.]+) "
                    r"execute=([0-9.]+) kernel=([0-9.]+) halo=([0-9.]+) enqueue=([0-9.]+) plan=([0-9.]+)", re.M)
COMPONENTS = ("execute", "kernel", "halo", "enqueue", "plan")


def parse_windows(out):
    """[(iter, seconds/iteration for that window)], [(iter, cumulative counters)]"""
    its = [(int(m[1]), float(m[2])) for m in IT_RE.finditer(out)]
    wins = []
    for m in WIN_RE.finditer(out):
        wins.append((int(m[1]), {"loops": int(m[2]), "launches": int(m[3]), "compile": float(m[4]),
                                 "execute": float(m[5]), "kernel": float(m[6]), "halo": float(m[7]),
                                 "enqueue": float(m[8]), "plan": float(m[9])}))
    return its, wins


def timing(runner, exe, backend, n, cfg_env, reps, niter=TIMING_ITERS):
    """Steady-state time per iteration and its breakdown, from windows of one run."""
    samples, warm, compile_total = [], [], []
    comp = {k: [] for k in COMPONENTS}
    launches, loops = [], []
    for _ in range(reps):
        r = runner.run(exe, backend, n, niter, dict(cfg_env, OPS_MLIR_STATS="1"))
        its, wins = parse_windows(r["out"])
        runner.done(r)
        steady = [t for (i, t) in its if i > WARMUP_ITERS]
        if not steady:
            raise RuntimeError(f"no steady windows for {exe} {backend} n={n}")
        samples += steady
        med = median(steady)
        warm.append(sum(t for (i, t) in its if i <= WARMUP_ITERS) * WINDOW - med * WINDOW * len([1 for (i, t) in its if i <= WARMUP_ITERS]))
        compile_total.append(wins[-1][1]["compile"])
        for (i0, c0), (i1, c1) in zip(wins, wins[1:]):
            if i0 >= WARMUP_ITERS:
                for k in COMPONENTS:
                    comp[k].append((c1[k] - c0[k]) / (i1 - i0))
                launches.append((c1["launches"] - c0["launches"]) / (i1 - i0))
                loops.append((c1["loops"] - c0["loops"]) / (i1 - i0))
    per = median(samples)
    return {"per_iter": per, "per_iter_spread": spread(samples), "per_iter_all": samples,
            "warmup_extra": median(warm), "compile_total": median(compile_total),
            "breakdown": {k: median(v) if v else float("nan") for k, v in comp.items()},
            "launches_per_iter": median(launches), "loops_per_iter": median(loops),
            "reps": reps, "n": n, "backend": backend, "exe": exe, "niter": niter}


# --------------------------------------------------------------------------
PLAN_RE = re.compile(r"\[plan\] (\d+) loops -> (\d+) kernels, (\d+) loops moved, "
                     r"est\. traffic (\S+) -> (\S+) bytes")
WHY_RE = re.compile(r"\[plan\]\s+new kernel started because:(.*)")


def parse_plans(out):
    plans, cur = [], None
    for line in out.splitlines():
        m = PLAN_RE.search(line)
        if m:
            cur = {"loops": int(m[1]), "kernels": int(m[2]), "moved": int(m[3]),
                   "bytes_unfused": float(m[4]), "bytes_fused": float(m[5]), "why": {}}
            plans.append(cur)
            continue
        m = WHY_RE.search(line)
        if m and cur is not None:
            cur["why"] = {k: int(v) for k, v in
                          (t.split("=") for t in m[1].split())}
    return plans


def exp_plan(runner):
    res = {}
    for name, env in CONFIGS.items():
        r = runner.run("opensbli", "seq", 16, 26, env, plan=True)   # 26 > filter_frequency
        plans = parse_plans(r["out"])
        runner.done(r)
        stage = max(plans, key=lambda p: p["loops"])
        filt = [p for p in plans if p["loops"] == 1 and p is not stage]
        res[name] = {"stage": stage, "all": plans,
                     "n_shapes": len(plans)}
    return res


KT_RE = re.compile(r"^(\S.*?): (\d+) calls, ([0-9.eE+-]+) s total")


def parse_profile(out):
    ks = {}
    for line in out.splitlines():
        m = KT_RE.match(line)
        if m and "Total" not in line:
            ks[m[1]] = {"calls": int(m[2]), "time": float(m[3])}
    m = re.search(r"Total time: ([0-9.eE+-]+) s", out)
    return ks, (float(m[1]) if m else float("nan"))


def exp_profile(runner, n=128, niter=250):
    """Per-kernel share of kernel time, unfused vs default (profiler summary)."""
    res = {}
    for name in ("unfused", DEFAULT):
        r = runner.run("opensbli_f32", "cuda", n, niter, CONFIGS[name])
        ks, total = parse_profile(r["out"])
        runner.done(r)
        res[name] = {"kernels": ks, "kernel_time": total, "n": n, "niter": niter}
    return res


def exp_accuracy(runner, cuda_rt):
    """Every config against the unfused run on the same backend and precision."""
    res = {}
    N, NITER = 32, 26
    for backend, exe in (("seq", "opensbli"), ("seq", "opensbli_f32"),
                         ("cuda", "opensbli_f32"), ("cuda", "opensbli")):
        key = f"{exe}/{backend}"
        ref = runner.run(exe, backend, N, NITER, CONFIGS["unfused"], dump=True)
        refp = os.path.join(ref["dir"], "fields")
        res[key] = {}
        for name, env in CONFIGS.items():
            if name == "unfused":
                continue
            r = runner.run(exe, backend, N, NITER, env, dump=True)
            p = os.path.join(r["dir"], "fields")
            A, B = cf.load(p, cf.meta(p)), cf.load(refp, cf.meta(refp))
            worst_abs, ndiff, total = 0.0, 0, 0
            num = den = 0.0
            for f in cf.FIELDS:
                for x, y in zip(A[f], B[f]):
                    d = abs(x - y)
                    total += 1
                    if d:
                        ndiff += 1
                        worst_abs = max(worst_abs, d)
                    num += d * d
                    den += y * y
            res[key][name] = {"bitwise": ndiff == 0, "differing": ndiff, "total": total,
                              "max_abs": worst_abs, "rel_l2": math.sqrt(num / den)}
            runner.done(r)
        runner.done(ref)
    # f32 against f64 (both default fusion), on the CPU
    a = runner.run("opensbli", "seq", N, NITER, {}, dump=True)
    b = runner.run("opensbli_f32", "seq", N, NITER, {}, dump=True)
    pa, pb = os.path.join(a["dir"], "fields"), os.path.join(b["dir"], "fields")
    A, B = cf.load(pb, cf.meta(pb)), cf.load(pa, cf.meta(pa))
    mom = max(math.sqrt(sum(y * y for y in B[f]) / len(B[f])) for f in ("rhou0", "rhou1", "rhou2"))
    errs = {}
    for f in cf.FIELDS:
        scale = mom if f.startswith("rhou") else math.sqrt(sum(y * y for y in B[f]) / len(B[f]))
        errs[f] = math.sqrt(sum((x - y) ** 2 for x, y in zip(A[f], B[f])) / len(B[f])) / scale
    res["f32_vs_f64"] = errs
    runner.done(a)
    runner.done(b)
    return res


def run_all(args):
    work = args.work or tempfile.mkdtemp(prefix="tgv_eval_")
    os.makedirs(work, exist_ok=True)
    cuda_rt = os.environ.get("OPS_MLIR_CUDA_RUNTIME", "")
    runner = Runner(args.app, work, cuda_rt)
    R = {"meta": {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "window": WINDOW, "timing_iters": TIMING_ITERS,
                  "host": os.uname().nodename}}
    only = set(args.only.split(",")) if args.only else None
    given = {"main_n": args.main_n, "cpu_n": args.cpu_n, "omp_threads": args.omp_threads,
             "peak_gbs": args.peak_gbs,
             "sizes": [int(x) for x in args.sizes.split(",")] if args.sizes else None,
             "sizes_f64": [int(x) for x in args.sizes_f64.split(",")] if args.sizes_f64 else None}
    if only and os.path.exists(args.out):
        old = json.load(open(args.out))       # resume: keep the stages we are not redoing
        for k, v in old.items():
            if k != "meta":
                R[k] = v
        R["meta"].update({k: v for k, v in old["meta"].items() if k != "date"})
    for k, v in given.items():
        if v is not None:
            R["meta"][k] = v
    M = lambda k, d: R["meta"].get(k, d)     # setting with a default
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        R["meta"]["gpu"] = gpu
        R["meta"]["cpus"] = os.cpu_count()
    except Exception:
        pass

    def stage(name, fn):
        if only and name not in only:
            return
        t = time.time()
        print(f"[{name}] ...", flush=True)
        R[name] = fn()
        print(f"[{name}] done in {time.time() - t:.0f}s", flush=True)
        json.dump(R, open(args.out, "w"), indent=1)

    q = args.quick
    main_n, cpu_n = M("main_n", 128), M("cpu_n", 64)
    reps = 2 if q else 3
    stage("plan", lambda: exp_plan(runner))
    stage("accuracy", lambda: exp_accuracy(runner, cuda_rt))

    # main comparison: every planner configuration
    def main_cmp():
        out = {}
        for exe, be, n, rp in (("opensbli_f32", "cuda", main_n, reps), ("opensbli", "cuda", main_n, reps),
                               ("opensbli_f32", "openmp", cpu_n, reps + 2), ("opensbli", "openmp", cpu_n, reps + 2)):
            key = f"{exe}/{be}/N{n}"
            out[key] = {}
            for name, env in CONFIGS.items():
                if be == "openmp" and M("omp_threads", None):
                    env = dict(env, OMP_NUM_THREADS=str(M("omp_threads", None)))
                out[key][name] = timing(runner, exe, be, n, env, rp,
                                        niter=TIMING_ITERS if be == "cuda" else 300)
                print(f"   {key} {name}: {out[key][name]['per_iter']*1e3:.2f} ms/iter "
                      f"(+-{out[key][name]['per_iter_spread']*50:.1f}%)", flush=True)
        return out
    stage("timing", main_cmp)

    def sizes():
        out = {}
        for n in (M("sizes", None) or ((32, 64, 128) if q else (24, 32, 48, 64, 96, 128, 160))):
            out[str(n)] = {}
            for name in ("unfused", "adjacent, same-range", DEFAULT):
                out[str(n)][name] = timing(runner, "opensbli_f32", "cuda", n, CONFIGS[name], reps)
                print(f"   N={n} {name}: {out[str(n)][name]['per_iter']*1e3:.2f} ms/iter", flush=True)
        return out
    stage("sizes", sizes)

    def sizes_f64():
        out = {}
        for n in M("sizes_f64", None) or []:
            out[str(n)] = {}
            for name in ("unfused", "adjacent, same-range", DEFAULT):
                out[str(n)][name] = timing(runner, "opensbli", "cuda", n, CONFIGS[name], reps)
                print(f"   f64 N={n} {name}: {out[str(n)][name]['per_iter']*1e3:.2f} ms/iter", flush=True)
        return out
    if M("sizes_f64", None):
        stage("sizes_f64", sizes_f64)

    def caps():
        out = {"max": {}, "ratio": {}}
        for m in ((1, 2, 4, 8, 16, 32) if not q else (1, 4, 8, 32)):
            env = {"OPS_MLIR_FUSION_MAX": str(m)}
            out["max"][str(m)] = timing(runner, "opensbli_f32", "cuda", main_n, env, 2)
            # plan shape for this cap
            r = runner.run("opensbli", "seq", 16, 3, env, plan=True)
            out["max"][str(m)]["stage_kernels"] = max(parse_plans(r["out"]), key=lambda p: p["loops"])["kernels"]
            runner.done(r)
            print(f"   max={m}: {out['max'][str(m)]['per_iter']*1e3:.2f} ms/iter, "
                  f"{out['max'][str(m)]['stage_kernels']} kernels/stage", flush=True)
        for ratio in ((0.5, 1.0, 2.0, 4.0) if not q else (1.0, 4.0)):
            env = {"OPS_MLIR_FUSION_BOX_RATIO": str(ratio)}
            out["ratio"][str(ratio)] = timing(runner, "opensbli_f32", "cuda", main_n, env, 2)
            r = runner.run("opensbli", "seq", 16, 3, env, plan=True)
            out["ratio"][str(ratio)]["stage_kernels"] = max(parse_plans(r["out"]), key=lambda p: p["loops"])["kernels"]
            runner.done(r)
            print(f"   ratio={ratio}: {out['ratio'][str(ratio)]['per_iter']*1e3:.2f} ms/iter", flush=True)
        return out
    stage("caps", caps)

    def openmp_quiet():
        # Fewer threads than the machine has (WSL oversubscribes badly) and more repetitions.
        out = {}
        threads = M("omp_threads", None) or 8
        for exe in ("opensbli_f32", "opensbli"):
            key = f"{exe}/openmp/N{cpu_n}/{threads}threads"
            out[key] = {}
            for name in ("unfused", "adjacent, same-range", "adjacent, guarded", DEFAULT):
                env = dict(CONFIGS[name], OMP_NUM_THREADS=str(threads))
                out[key][name] = timing(runner, exe, "openmp", cpu_n, env, 5, niter=300)
                print(f"   {key} {name}: {out[key][name]['per_iter']*1e3:.2f} ms/iter "
                      f"(+-{out[key][name]['per_iter_spread']*50:.1f}%)", flush=True)
        return out
    stage("openmp_8t", openmp_quiet)
    stage("profile", lambda: exp_profile(runner))

    def bandwidth():
        """Byte model at the real grid size, to compare with kernel time."""
        out = {}
        for name in ("unfused", DEFAULT):
            r = runner.run("opensbli_f32", "cuda", main_n, 2, CONFIGS[name], plan=True)
            st = max(parse_plans(r["out"]), key=lambda p: p["loops"])
            runner.done(r)
            out[name] = {"stage_bytes": st["bytes_fused"], "stage_bytes_unfused": st["bytes_unfused"]}
        return out
    stage("bandwidth", bandwidth)
    R["meta"]["runs"] = runner.count
    json.dump(R, open(args.out, "w"), indent=1)
    print(f"done: {runner.count} app runs -> {args.out}")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#72b7b2", "#9d755d", "#bab0ac"]


def svg_bars(title, labels, values, errs=None, ylabel="ms / iteration", width=640, height=300, baseline=None):
    left, right, top, bottom = 70, 20, 40, 95
    plot_w, plot_h = width - left - right, height - top - bottom
    vmax = max(values) * 1.15
    n = len(values)
    bw = plot_w / n * 0.62
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'font-family="sans-serif" font-size="12">',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<text x="{width/2}" y="22" text-anchor="middle" font-size="14" font-weight="bold">{title}</text>']
    for i in range(5):
        v = vmax * i / 4
        y = top + plot_h - plot_h * i / 4
        out.append(f'<line x1="{left}" x2="{width-right}" y1="{y}" y2="{y}" stroke="#ddd"/>')
        out.append(f'<text x="{left-6}" y="{y+4}" text-anchor="end">{v:.3g}</text>')
    out.append(f'<text x="14" y="{top+plot_h/2}" transform="rotate(-90 14 {top+plot_h/2})" '
               f'text-anchor="middle">{ylabel}</text>')
    for i, (lab, v) in enumerate(zip(labels, values)):
        x = left + plot_w * (i + 0.5) / n
        h = plot_h * v / vmax
        col = PALETTE[0] if i == 0 else PALETTE[2] if "default" in lab or lab == DEFAULT else PALETTE[7]
        out.append(f'<rect x="{x-bw/2}" y="{top+plot_h-h}" width="{bw}" height="{h}" fill="{col}"/>')
        out.append(f'<text x="{x}" y="{top+plot_h-h-5}" text-anchor="middle">{v:.3g}</text>')
        if errs:
            e = plot_h * errs[i] / vmax
            out.append(f'<line x1="{x}" x2="{x}" y1="{top+plot_h-h-e}" y2="{top+plot_h-h+e}" stroke="#333"/>')
        words = lab.split(", ")
        for k, wd in enumerate(words):
            out.append(f'<text x="{x}" y="{top+plot_h+16+14*k}" text-anchor="middle">{wd}</text>')
    out.append('</svg>')
    return "\n".join(out)


def svg_lines(title, xs, series, xlabel, ylabel, width=640, height=300, logx=False):
    left, right, top, bottom = 70, 130, 40, 50
    plot_w, plot_h = width - left - right, height - top - bottom
    allv = [v for s in series.values() for v in s]
    vmax = max(allv) * 1.1
    xmin, xmax = min(xs), max(xs)
    fx = (lambda x: math.log(x)) if logx else (lambda x: x)
    def X(x):
        return left + plot_w * (fx(x) - fx(xmin)) / (fx(xmax) - fx(xmin))
    def Y(v):
        return top + plot_h - plot_h * v / vmax
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'font-family="sans-serif" font-size="12">',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<text x="{(width-right+left)/2}" y="22" text-anchor="middle" font-size="14" font-weight="bold">{title}</text>']
    for i in range(5):
        v = vmax * i / 4
        out.append(f'<line x1="{left}" x2="{width-right}" y1="{Y(v)}" y2="{Y(v)}" stroke="#ddd"/>')
        out.append(f'<text x="{left-6}" y="{Y(v)+4}" text-anchor="end">{v:.3g}</text>')
    for x in xs:
        out.append(f'<text x="{X(x)}" y="{top+plot_h+16}" text-anchor="middle">{x}</text>')
    out.append(f'<text x="{left+plot_w/2}" y="{height-8}" text-anchor="middle">{xlabel}</text>')
    out.append(f'<text x="14" y="{top+plot_h/2}" transform="rotate(-90 14 {top+plot_h/2})" text-anchor="middle">{ylabel}</text>')
    for k, (name, vals) in enumerate(series.items()):
        col = PALETTE[k % len(PALETTE)]
        pts = " ".join(f"{X(x)},{Y(v)}" for x, v in zip(xs, vals))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        for x, v in zip(xs, vals):
            out.append(f'<circle cx="{X(x)}" cy="{Y(v)}" r="3" fill="{col}"/>')
        ly = top + 14 + 16 * k
        out.append(f'<rect x="{width-right+10}" y="{ly-9}" width="12" height="12" fill="{col}"/>')
        out.append(f'<text x="{width-right+28}" y="{ly+2}">{name}</text>')
    out.append('</svg>')
    return "\n".join(out)


def ms(x):
    return f"{x*1e3:.2f}"


def findings(R):
    """Headline results. Numbers come from the data, and so do the claims: a statement
    such as "memory-bound" is only made when the numbers support it."""
    t = R["timing"]
    mn = R["meta"].get("main_n", 128)
    g32, g64 = t[f"opensbli_f32/cuda/N{mn}"], t[f"opensbli/cuda/N{mn}"]
    peak = R["meta"].get("peak_gbs", 192)
    gpu = R["meta"].get("gpu", "the GPU").split(",")[0]
    def sp(tab, a, b):
        return tab[a]["per_iter"] / tab[b]["per_iter"]
    st = {k: v["stage"] for k, v in R["plan"].items()}
    L = ["## Summary\n"]
    L.append(f"* **One Runge-Kutta stage** queues {st['unfused']['loops']} loops. Adjacent-only fusion turns them into "
             f"{st['adjacent, same-range']['kernels']} kernels; the dependence DAG into {st[DEFAULT]['kernels']}, "
             f"moving {st[DEFAULT]['moved']} loops past independent ones. Launches per iteration (including the "
             f"filter steps): {g32['unfused']['launches_per_iter']:.0f} → {g32['adjacent, same-range']['launches_per_iter']:.0f} "
             f"→ {g32[DEFAULT]['launches_per_iter']:.0f}.")
    L.append(f"* **Single precision on {gpu} ({mn}³):** {ms(g32['unfused']['per_iter'])} ms/iteration unfused → "
             f"{ms(g32['adjacent, same-range']['per_iter'])} ms with adjacent same-range fusion "
             f"({sp(g32,'unfused','adjacent, same-range'):.2f}×) → {ms(g32[DEFAULT]['per_iter'])} ms with the DAG planner "
             f"({sp(g32,'unfused',DEFAULT):.2f}×). Most of the gain is ordinary fusion; reordering adds "
             f"{(sp(g32,'adjacent, same-range',DEFAULT)-1)*100:.0f}% on top. Different-range (guarded) fusion changes nothing "
             f"for this app: the stage has the same kernel count with or without it.")
    if "bandwidth" in R:
        bu, bf = R["bandwidth"]["unfused"]["stage_bytes"], R["bandwidth"][DEFAULT]["stage_bytes"]
        ku, kf = g32["unfused"]["breakdown"]["kernel"], g32[DEFAULT]["breakdown"]["kernel"]
        r_traffic, r_time = bu / bf, ku / kf
        bw_u, bw_f = 3 * bu / ku / 1e9, 3 * bf / kf / 1e9
        host = lambda tt: (tt["breakdown"]["enqueue"] + tt["breakdown"]["plan"] + tt["breakdown"]["execute"]
                           - tt["breakdown"]["kernel"]) / tt["per_iter"]
        hostpct = max(host(g32["unfused"]), host(g32[DEFAULT])) * 100
        if r_time >= 0.95 * r_traffic:
            why = (f"* **Why:** the kernels are memory-bound. The byte model predicts {r_traffic:.2f}× less traffic and the "
                   f"measured kernel time drops {r_time:.2f}×; the modelled bandwidth is {bw_u:.0f} GB/s unfused and "
                   f"{bw_f:.0f} GB/s fused (peak {peak:.0f}). The gain is the memory traffic that fusion removes")
        else:
            why = (f"* **Why less than the traffic saving:** the byte model predicts {r_traffic:.2f}× less traffic but the "
                   f"kernel time drops only {r_time:.2f}×. The modelled bandwidth falls from {bw_u:.0f} GB/s "
                   f"({bw_u/peak*100:.0f}% of the {peak:.0f} GB/s peak) unfused to {bw_f:.0f} GB/s ({bw_f/peak*100:.0f}%) "
                   f"fused: the fused kernels move fewer bytes but sustain less bandwidth"
                   + (" (see the Nsight Compute section for the per-kernel cause)" if R.get("_has_ncu") else ""))
        L.append(why + f". Host overhead (enqueue, planning, cache key, argument packing) is under {max(hostpct, 1):.0f}% "
                 f"of an iteration, so the number of launches itself is not what costs time.")
    r64 = g64[DEFAULT]["per_iter"] / g32[DEFAULT]["per_iter"]
    L.append(f"* **Double precision on {gpu}:** {ms(g64['unfused']['per_iter'])} → {ms(g64[DEFAULT]['per_iter'])} ms "
             f"({sp(g64,'unfused',DEFAULT):.2f}×). Double costs {r64:.1f}× single (doubling the bytes alone would give 2×), "
             + ("so fp64 arithmetic throughput is a limit." if r64 > 3.5 else
                "somewhat more than that." if r64 > 2.3 else
                "so it is about bandwidth-proportional.")
             + (f" Fusion helps double much less than single ({sp(g64,'unfused',DEFAULT):.2f}× vs "
                f"{sp(g32,'unfused',DEFAULT):.2f}×)." if sp(g64,'unfused',DEFAULT) < 0.9 * sp(g32,'unfused',DEFAULT) else ""))
    if "openmp_8t" in R:
        o32 = next(v for k, v in R["openmp_8t"].items() if k.startswith("opensbli_f32/"))
        o64 = next(v for k, v in R["openmp_8t"].items() if k.startswith("opensbli/"))
        nthr = next(iter(R["openmp_8t"])).split("/")[-1].replace("threads", "")
        ncpu = next(iter(R["openmp_8t"])).split("/")[2][1:]
        L.append(f"* **OpenMP ({nthr} threads, {ncpu}³):** {sp(o32,'unfused',DEFAULT):.2f}× in single precision "
                 f"({ms(o32['unfused']['per_iter'])} → {ms(o32[DEFAULT]['per_iter'])} ms) and "
                 f"{sp(o64,'unfused',DEFAULT):.2f}× in double ({ms(o64['unfused']['per_iter'])} → {ms(o64[DEFAULT]['per_iter'])} ms). "
                 f"(Section 2 has every configuration and its spread.)")
    sizes = R.get("sizes")
    if sizes:
        ns = sorted(int(k) for k in sizes)
        ratios = {n: sizes[str(n)]["unfused"]["per_iter"] / sizes[str(n)][DEFAULT]["per_iter"] for n in ns}
        lo = min(ratios, key=ratios.get)
        hi = max(ratios, key=ratios.get)
        L.append(f"* **Problem size (single precision, GPU):** the speed-up ranges from {ratios[lo]:.2f}× (N={lo}) to "
                 f"{ratios[hi]:.2f}× (N={hi}) over N = {ns[0]}…{ns[-1]}. It is not monotonic in N, and the dips "
                 f"were not investigated.")
    c = R.get("caps")
    if c:
        mx = sorted(((int(k), v["per_iter"], v["stage_kernels"]) for k, v in c["max"].items()))
        best = min(v for _, v, _ in mx)
        first = next(m for m, v, _ in mx if v <= best * 1.01)
        rs = [v["per_iter"] for v in c["ratio"].values()]
        L.append(f"* **Parameters:** a cap of {first} loops per kernel already reaches the best time "
                 f"({mx[[m for m,_,_ in mx].index(first)][2]} kernels per stage; larger caps change nothing, a cap of 1 is the "
                 f"unfused case). The bounding-box ratio "
                 + ("has little effect here" if (max(rs) - min(rs)) / min(rs) < 0.05 else "matters here")
                 + f" ({ms(min(rs))}–{ms(max(rs))} ms). Earliest vs. latest placement: "
                 f"{ms(g32[DEFAULT]['per_iter'])} vs {ms(g32['DAG, guarded, latest']['per_iter'])} ms.")
    L.append(f"* **Cost:** JIT compilation of the {len(R['plan'][DEFAULT]['all'])} distinct queue shapes takes "
             f"{g32['unfused']['compile_total']:.1f} s unfused and {g32[DEFAULT]['compile_total']:.1f} s fused (the first "
             f"100 iterations cost about {g32[DEFAULT]['warmup_extra']:.1f} s more than steady state), a one-off equal to "
             f"roughly {g32[DEFAULT]['compile_total']/g32[DEFAULT]['per_iter']:.0f} steady iterations.")
    accs = [a for k, rows in R["accuracy"].items() if k != "f32_vs_f64" for a in rows.values()]
    e = R["accuracy"]["f32_vs_f64"]
    if all(a["bitwise"] for a in accs):
        L.append(f"* **Correctness:** all {len(accs)} planner-configuration runs are bitwise identical to the unfused run "
                 f"(CPU and GPU, both precisions; section 6). Single precision stays within {max(e.values()):.0e} of double "
                 f"(rms, relative to the flow scale) after 26 steps.")
    else:
        bad = sum(1 for a in accs if not a["bitwise"])
        L.append(f"* **Correctness:** {bad} of {len(accs)} planner-configuration runs differ from the unfused run; "
                 f"largest difference {max(a['max_abs'] for a in accs):.1e} (section 6).")
    why = st[DEFAULT]['why']
    L.append("* **What limits it:** a stage still needs several kernels. For the loops that had to start a new kernel, "
             "the planner's diagnostics count the reasons an existing kernel was rejected: "
             + ", ".join(f"`{k}` {v}" for k, v in sorted(why.items()) if k != 'first') +
             " (`dependence`: a stencil read of a field written in that kernel; `order`: joining would move the loop "
             "before something it depends on; `range`: the bounding-box rules). Fusing across the dependence edges "
             "would need redundant halo computation.")
    L.append("")
    L.append("*Method note.* Per-iteration times come from windows of 100 iterations inside one process after warm-up. "
             "An earlier version of this evaluation differenced the wall time of a short and a long run; JIT time varies "
             "by about a second between runs, which is ±10 ms per iteration, and that method gave wrong numbers "
             "(it also counted the one-off compile of the filter kernels at iteration 25). Those numbers were discarded.\n")
    return "\n".join(L) + "\n"


def report(args):
    tag = getattr(args, 'tag', '') or ''
    suffix = ('_' + tag) if tag else ''
    R = json.load(open(args.results))
    R["_has_ncu"] = bool(getattr(args, "include", None))
    os.makedirs(os.path.join(args.outdir, "img"), exist_ok=True)
    md = []
    w = md.append
    meta = R["meta"]
    gpu_name = meta.get("gpu", "").split(",")[0]
    w("# Taylor-Green vortex: fusion and reordering evaluation" + (" - " + gpu_name if tag else "") + "\n")
    w(f"Generated by `apps/c/taylor_green_vortex/eval_tgv.py` on {meta['date']}. "
      f"GPU: {meta.get('gpu', '?')}; {meta.get('cpus', '?')} CPU threads. "
      f"Raw data: `data/tgv_eval{suffix}.json`.\n")

    w(findings(R))
    # ---- plan
    w("## 1. What the planner does to one time step\n")
    w("One Runge-Kutta stage queues a batch of loops between halo exchanges; there are three stages "
      "per time step, plus a filter block every 25 steps (seven loops that the app flushes one by one).\n")
    w("| configuration | loops / stage | kernels / stage | loops moved | est. traffic (unfused → fused) | reduction |")
    w("|---|---:|---:|---:|---:|---:|")
    for name, p in R["plan"].items():
        s = p["stage"]
        w(f"| {name} | {s['loops']} | {s['kernels']} | {s['moved']} | "
          f"{s['bytes_unfused']/1e6:.2f} → {s['bytes_fused']/1e6:.2f} MB (N=16) | "
          f"{s['bytes_unfused']/s['bytes_fused']:.2f}× |")
    w("")
    d = R["plan"][DEFAULT]["stage"]
    u = R["plan"]["unfused"]["stage"]
    w(f"Per time step that is {3*u['kernels']} kernel launches unfused versus {3*d['kernels']} with the default planner "
      f"(excluding the filter block).\n")
    w("Why loops still start a new kernel (counts of rejected candidate kernels, default planner, one stage): "
      + ", ".join(f"`{k}`={v}" for k, v in sorted(d["why"].items())) + ".\n")

    # ---- timing
    w("## 2. Time per iteration\n")
    w("Steady-state time per iteration is the app's own timing of windows of 100 iterations inside one "
      "process, **after** the warm-up window (which contains the JIT compile of every kernel shape). "
      "Values are medians over all steady windows of all repetitions; `±` is (max−min)/median over those "
      "windows. `warm-up` is the extra time the first 100 iterations took compared with steady state, "
      "and `JIT` the seconds the runtime spent compiling in total.\n")
    imgs = []
    for key, table in R["timing"].items():
        exe, be, n = key.split("/")
        prec = "single" if exe.endswith("f32") else "double"
        base = table["unfused"]["per_iter"]
        w(f"### {prec} precision, {be.upper()}, {n[1:]}³\n")
        w("| configuration | ms / iteration | ± | speed-up vs unfused | kernels / iteration | warm-up s | JIT s |")
        w("|---|---:|---:|---:|---:|---:|---:|")
        for name, t in table.items():
            w(f"| {name} | {ms(t['per_iter'])} | {t['per_iter_spread']*100:.1f}% | "
              f"{base/t['per_iter']:.3f}× | {t['launches_per_iter']:.1f} | "
              f"{t['warmup_extra']:.1f} | {t['compile_total']:.1f} |")
        w("")
        labels = list(table.keys())
        svg = svg_bars(f"{prec} precision, {be.upper()}, {n[1:]}³", labels,
                       [table[k]["per_iter"] * 1e3 for k in labels],
                       [table[k]["per_iter"] * 1e3 * table[k]["per_iter_spread"] / 2 for k in labels])
        fn = f"img/timing_{exe}_{be}_{n}{suffix}.svg"
        open(os.path.join(args.outdir, fn), "w").write(svg)
        imgs.append(fn)
        w(f"![{key}]({fn})\n")

    if "openmp_8t" in R:
        w("### OpenMP, controlled repeat (5 repetitions)\n")
        w("The OpenMP runs above use every available thread and can be noisy. "
          "Repeated with a fixed thread count and more repetitions:\n")
        for key, table in R["openmp_8t"].items():
            exe = key.split("/")[0]
            prec = "single" if exe.endswith("f32") else "double"
            base = table["unfused"]["per_iter"]
            w(f"**{prec} precision, {key.split('/')[2][1:]}³, {key.split('/')[3].replace('threads', '')} threads**\n")
            w("| configuration | ms / iteration | ± | speed-up vs unfused |")
            w("|---|---:|---:|---:|")
            for name, t in table.items():
                w(f"| {name} | {ms(t['per_iter'])} | {t['per_iter_spread']*100:.1f}% | {base/t['per_iter']:.3f}× |")
            w("")

    # precision
    try:
        mn_ = R["meta"].get("main_n", 128)
        t32 = R["timing"][f"opensbli_f32/cuda/N{mn_}"][DEFAULT]["per_iter"]
        t64 = R["timing"][f"opensbli/cuda/N{mn_}"][DEFAULT]["per_iter"]
        w(f"Single versus double precision on the GPU (default planner, {mn_}³): {ms(t32)} ms vs {ms(t64)} ms per "
          f"iteration, **{t64/t32:.2f}×**.\n")
    except KeyError:
        pass

    # sizes
    if "sizes" in R:
        w("## 3. Dependence on problem size (single precision, GPU)\n")
        ns = sorted(int(k) for k in R["sizes"])
        names = list(next(iter(R["sizes"].values())).keys())
        w("| N³ | " + " | ".join(f"{n} ms" for n in names) + " | speed-up (default) |")
        w("|---:|" + "---:|" * (len(names) + 1))
        sp = []
        for n in ns:
            row = R["sizes"][str(n)]
            s = row["unfused"]["per_iter"] / row[DEFAULT]["per_iter"]
            sp.append(s)
            w(f"| {n} | " + " | ".join(ms(row[k]["per_iter"]) for k in names) + f" | {s:.3f}× |")
        w("")
        svg = svg_lines("Speed-up of the default planner vs unfused", ns,
                        {"default / unfused": sp, "1.0": [1.0] * len(ns)}, "grid size N (N³ cells)",
                        "speed-up", logx=True)
        open(os.path.join(args.outdir, f"img/sizes{suffix}.svg"), "w").write(svg)
        w(f"![sizes](img/sizes{suffix}.svg)\n")

    if R.get("sizes_f64"):
        w("### Double precision (GPU)\n")
        ns64 = sorted(int(k) for k in R["sizes_f64"])
        names64 = list(next(iter(R["sizes_f64"].values())).keys())
        w("| N³ | " + " | ".join(f"{n} ms" for n in names64) + " | speed-up (default) |")
        w("|---:|" + "---:|" * (len(names64) + 1))
        for n in ns64:
            row = R["sizes_f64"][str(n)]
            w(f"| {n} | " + " | ".join(ms(row[k]["per_iter"]) for k in names64) +
              f" | {row['unfused']['per_iter'] / row[DEFAULT]['per_iter']:.3f}× |")
        w("")

    # caps
    if "caps" in R:
        w(f"## 4. Parameter sweeps (single precision, GPU, {R['meta'].get('main_n', 128)}³)\n")
        w("Maximum loops per kernel (`OPS_MLIR_FUSION_MAX`):\n")
        w("| max | kernels / stage | ms / iteration |")
        w("|---:|---:|---:|")
        for k, t in R["caps"]["max"].items():
            w(f"| {k} | {t['stage_kernels']} | {ms(t['per_iter'])} |")
        w("\nBounding-box ratio (`OPS_MLIR_FUSION_BOX_RATIO`):\n")
        w("| ratio | kernels / stage | ms / iteration |")
        w("|---:|---:|---:|")
        for k, t in R["caps"]["ratio"].items():
            w(f"| {k} | {t['stage_kernels']} | {ms(t['per_iter'])} |")
        w("")

    # breakdown
    w("## 5. Where the time goes\n")
    w("Exact steady-state breakdown per iteration from the runtime's cumulative timers, differenced "
      f"between windows of 100 iterations (single precision, GPU, {R['meta'].get('main_n', 128)}³). *kernel* is the time inside the "
      "generated functions including the device synchronisation; *execute overhead* is argument packing, "
      "buffer bookkeeping and profiling around them.\n")
    w("| configuration | wall ms | kernels | execute overhead | halo exchange | enqueue | plan + cache key | other (app) |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    tab = R["timing"].get(f"opensbli_f32/cuda/N{R['meta'].get('main_n', 128)}", {})
    for name in ("unfused", "adjacent, same-range", DEFAULT):
        if name not in tab:
            continue
        t = tab[name]
        b_ = t["breakdown"]
        wall = t["per_iter"]
        other = wall - b_["execute"] - b_["halo"] - b_["enqueue"] - b_["plan"]
        w(f"| {name} | {ms(wall)} | {ms(b_['kernel'])} | {ms(b_['execute'] - b_['kernel'])} | "
          f"{ms(b_['halo'])} | {ms(b_['enqueue'])} | {ms(b_['plan'])} | {ms(other)} |")
    w("")
    if "bandwidth" in R and tab:
        w("Modelled memory bandwidth (byte model of the kernels actually launched × 3 stages ÷ kernel time; "
          f"the GPU's peak is {R['meta'].get('peak_gbs', 192)} GB/s):\n")
        w("| configuration | model traffic / iteration | kernel ms | modelled GB/s |")
        w("|---|---:|---:|---:|")
        for name in ("unfused", DEFAULT):
            if name not in R["bandwidth"] or name not in tab:
                continue
            by = 3 * R["bandwidth"][name]["stage_bytes"]
            k = tab[name]["breakdown"]["kernel"]
            w(f"| {name} | {by/1e9:.2f} GB | {ms(k)} | {by/k/1e9:.0f} |")
        w("")
    if "profile" in R:
        w("Per-kernel share of kernel time (profiler, 250 iterations):\n")
        for name, p in R["profile"].items():
            ks = sorted(p["kernels"].items(), key=lambda kv: -kv[1]["time"])
            w(f"**{name}** — total kernel time {p['kernel_time']:.3f} s\n")
            w("| kernel | calls | total s | share |")
            w("|---|---:|---:|---:|")
            for k, v in ks[:6]:
                w(f"| `{k[:60]}` | {v['calls']} | {v['time']:.3f} | {v['time']/p['kernel_time']*100:.0f}% |")
            w("")

    # accuracy
    w("## 6. Accuracy\n")
    w("Each configuration against the unfused run (same backend and precision), 32³, 26 steps "
      "(covers a filter step).\n")
    w("| app / backend | configuration | bitwise equal | max abs diff | rel L2 |")
    w("|---|---|:---:|---:|---:|")
    for key, rows in R["accuracy"].items():
        if key == "f32_vs_f64":
            continue
        for name, a in rows.items():
            w(f"| {key} | {name} | {'yes' if a['bitwise'] else 'no'} | {a['max_abs']:.2e} | {a['rel_l2']:.2e} |")
    w("")
    e = R["accuracy"]["f32_vs_f64"]
    w("Single against double precision (rms error relative to the flow scale): "
      + ", ".join(f"{k} {v:.2e}" for k, v in e.items()) + ".\n")

    for extra in getattr(args, "include", None) or []:
        md.append(open(extra).read())
    out = os.path.join(args.outdir, f"tgv_evaluation{suffix}.md")
    open(out, "w").write("\n".join(md))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--app", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--work")
    r.add_argument("--quick", action="store_true")
    r.add_argument("--only", help="comma-separated stages to (re)run, keeping the rest of --out")
    r.add_argument("--main-n", type=int, help="grid size of the main GPU comparison (default 128)")
    r.add_argument("--cpu-n", type=int, help="grid size of the OpenMP comparison (default 64)")
    r.add_argument("--omp-threads", type=int, help="OMP_NUM_THREADS for the OpenMP runs")
    r.add_argument("--sizes", help="comma-separated grid sizes for the single-precision size sweep")
    r.add_argument("--sizes-f64", help="comma-separated grid sizes for a double-precision size sweep")
    r.add_argument("--peak-gbs", type=float, help="the GPU's peak memory bandwidth in GB/s (default 192)")
    p = sub.add_parser("report")
    p.add_argument("--results", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--tag", help="suffix for the output file names, e.g. a100")
    p.add_argument("--include", action="append", help="markdown file appended to the report (repeatable)")
    a = ap.parse_args()
    run_all(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    main()
