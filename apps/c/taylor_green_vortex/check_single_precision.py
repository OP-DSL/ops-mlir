#!/usr/bin/env python3
"""Verify the generated single-precision Taylor-Green vortex really is single
precision, with no implicit promotion to double.

    check_single_precision.py <src_dir> <generated_dir> <kernel-ir-dump>

1. Static: in every converted file there is no `double` (except host timers),
   no floating literal without an `f` suffix, no M_PI, no "double" type
   string, and no bare <math.h> call.
2. JIT level: every kernel in the generated header is translated by
   KernelIRBuilder with OPS_MLIR_STRICT_FP=1 (which rejects a float-interface
   kernel that computes in double), and the resulting IR contains no f64.
3. Negative control: a deliberately naive conversion (`double`->`float` only)
   must be rejected by the same strict translation, so this test cannot pass
   by the detector simply being blind.
Prints PASS/FAIL lines and exits non-zero on any failure.
"""
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import to_single_precision as conv  # noqa: E402

failures = 0


def report(ok, name, detail=""):
    global failures
    print(("[PASS] " if ok else "[FAIL] ") + name + ("" if ok else ": " + detail))
    if not ok:
        failures += 1


def static_problems(path):
    text = open(path).read()
    toks = conv.tokenize(text)
    problems = []
    for i, (kind, t) in enumerate(toks):
        if kind == "num" and conv.is_float_literal(t) and t[-1] not in "fFlL":
            problems.append(f"unsuffixed float literal {t}")
        elif kind == "str" and t == '"double"':
            problems.append('"double" type string')
        elif kind == "id":
            if t == "M_PI":
                problems.append("M_PI (double)")
            elif t == "double":
                j = i + 1
                while j < len(toks) and (toks[j][0] == "ws" or toks[j][1] in ("*", "&", "const")):
                    j += 1
                if not conv.KEEP_DOUBLE.match(toks[j][1] if j < len(toks) else ""):
                    problems.append("double declaration of " + toks[j][1])
            elif t in conv.MATH:
                nxt = next((x for x in toks[i + 1:] if x[0] != "ws"), ("", ""))
                if nxt[1] == "(":
                    problems.append(f"bare math call {t}(")
    return problems


def kernel_interface(header):
    text = open(header).read()
    floats = re.findall(r"^extern float (\w+);", text, re.M)
    ints = re.findall(r"^extern int (\w+);", text, re.M)
    doubles = re.findall(r"^extern double (\w+);", text, re.M)
    kernels = re.findall(r"^(?:void|float|double) (opensbliblock00Kernel\d+)\s*\(", text, re.M)
    return floats, doubles, ints, kernels


def translate(dump, header, kernel, floats, doubles, ints):
    cmd = [dump, header, kernel, "3"]
    for n in floats:
        cmd += ["-float", f"{n}=0.5"]
    for n in doubles:
        cmd += ["-double", f"{n}=0.5"]
    for n in ints:
        cmd += ["-int", f"{n}=8"]
    env = dict(os.environ, OPS_MLIR_STRICT_FP="1")
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def main():
    src, gen, dump = sys.argv[1:4]

    # 1. static checks on the converted files
    for name in conv.CONVERT:
        problems = static_problems(os.path.join(gen, name))
        report(not problems, "static/" + name, "; ".join(sorted(set(problems))[:5]))

    # 2. every generated kernel, strictly
    header = os.path.join(gen, "opensbliblock00_kernels.h")
    floats, doubles, ints, kernels = kernel_interface(header)
    report(len(kernels) > 20 and not doubles, "generated header exposes kernels",
           f"{len(kernels)} kernels, {len(doubles)} double externs")
    bad = []
    for k in kernels:
        r = translate(dump, header, k, floats, doubles, ints)
        if r.returncode != 0 or "f64" in r.stdout:
            bad.append(k + ": " + (r.stderr.strip().splitlines() or ["f64 in IR"])[-1][:140])
    report(not bad, f"strict translation of {len(kernels)} kernels, IR has no f64",
           "; ".join(bad[:3]))

    # 3. negative control: naive double->float only
    with tempfile.TemporaryDirectory() as tmp:
        text = open(os.path.join(src, "opensbliblock00_kernels.h")).read()
        naive = re.sub(r"\bdouble\b", "float", text)
        naive_header = os.path.join(tmp, "naive_kernels.h")
        open(naive_header, "w").write(naive)
        n_floats, n_doubles, n_ints, n_kernels = kernel_interface(naive_header)
        rejected = [k for k in n_kernels
                    if translate(dump, naive_header, k, n_floats, n_doubles, n_ints).returncode != 0]
        report(len(rejected) > 0, "negative control: naive conversion is rejected",
               "strict mode accepted every kernel of a conversion that leaves double literals in")
        print(f"       ({len(rejected)}/{len(n_kernels)} naive kernels rejected)")

    print(f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
