#!/usr/bin/env python3
"""Run the double- and single-precision TGV apps on a small grid and check the
single-precision fields stay within single-precision distance of the double
ones (rms error per field, relative to the flow's scale), with no NaNs.

    check_tgv_precision.py <app_dir> <backend> [--tol 1e-4]
"""
import math
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_fields as cf  # noqa: E402


def run(exe, backend, tmp, tag):
    d = os.path.join(tmp, tag)
    os.makedirs(d)
    env = dict(os.environ, OPS_BACKEND=backend, TGV_N="16", TGV_NITER="20",
               TGV_DIAG="1", TGV_DUMP=os.path.join(d, "fields"),
               OPS_MLIR_STRICT_FP="1")
    r = subprocess.run([exe], cwd=d, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print("FAIL", tag, "exit", r.returncode, r.stderr[-300:])
        sys.exit(1)
    return os.path.join(d, "fields"), r.stdout


def main():
    app_dir, backend = sys.argv[1:3]
    tol = float(sys.argv[sys.argv.index("--tol") + 1]) if "--tol" in sys.argv else 1e-4
    with tempfile.TemporaryDirectory() as tmp:
        p64, out64 = run(os.path.join(app_dir, "opensbli"), backend, tmp, "f64")
        p32, out32 = run(os.path.join(app_dir, "opensbli_f32"), backend, tmp, "f32")
        if "nan=0" not in out32 or "nan=0" not in out64:
            print("FAIL: NaNs in output")
            return 1
        A = cf.load(p32, cf.meta(p32))
        B = cf.load(p64, cf.meta(p64))
        ok = True

        def rms(v):
            return math.sqrt(sum(y * y for y in v) / len(v))

        # Each momentum component is measured against the rms momentum of the
        # flow, not its own norm: rhou2 is ~0 early in the Taylor-Green
        # vortex, so its own relative error would divide roundoff by roundoff.
        mom_scale = max(rms(B[f]) for f in ("rhou0", "rhou1", "rhou2"))
        for f in cf.FIELDS:
            scale = mom_scale if f.startswith("rhou") else rms(B[f])
            err = math.sqrt(sum((x - y) ** 2 for x, y in zip(A[f], B[f])) / len(B[f]))
            rel = err / scale
            good = rel <= tol
            ok &= good
            print(("PASS" if good else "FAIL"),
                  f"{f}: rms error / flow scale (f32 vs f64) = {rel:.3e} (tol {tol:.0e})")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
