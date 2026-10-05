#!/usr/bin/env python3
"""Compare two raw field dumps written by the TGV app (TGV_DUMP=<prefix>).

    compare_fields.py <prefixA> <prefixB> [--bitwise]

Prints, per field, max |A-B|, the relative L2 error ||A-B|| / ||B|| and the
number of values that differ. Precision of each dump comes from its .meta
file (sizeof), so f32 and f64 runs can be compared directly: values are
compared in double. With --bitwise exits non-zero unless every value is
identical. No numpy needed.
"""
import array
import math
import sys

FIELDS = ["rho", "rhou0", "rhou1", "rhou2", "rhoE"]


def meta(prefix):
    d = {}
    for line in open(prefix + ".meta"):
        k, v = line.strip().split("=")
        d[k] = v
    return d


def load(prefix, m):
    a = array.array("f" if int(m["sizeof"]) == 4 else "d")
    out = {}
    for f in FIELDS:
        a = array.array("f" if int(m["sizeof"]) == 4 else "d")
        with open(f"{prefix}_{f}.bin", "rb") as fh:
            a.frombytes(fh.read())
        out[f] = a
    return out


def main():
    pa, pb = sys.argv[1:3]
    bitwise = "--bitwise" in sys.argv
    ma, mb = meta(pa), meta(pb)
    A, B = load(pa, ma), load(pb, mb)
    print(f"A: {pa} (sizeof={ma['sizeof']}, iter={ma['iter']})   "
          f"B: {pb} (sizeof={mb['sizeof']}, iter={mb['iter']})")
    worst_diff = 0
    for f in FIELDS:
        a, b = A[f], B[f]
        assert len(a) == len(b), "size mismatch"
        maxd = 0.0
        num = den = 0.0
        ndiff = 0
        for x, y in zip(a, b):
            d = abs(x - y)
            if d:
                ndiff += 1
                if d > maxd:
                    maxd = d
            num += d * d
            den += y * y
        rel = math.sqrt(num / den) if den else float("nan")
        worst_diff += ndiff
        print(f"  {f:6s} max|A-B|={maxd:.3e}  relL2={rel:.3e}  differing={ndiff}/{len(a)}")
    if bitwise and worst_diff:
        print("NOT bitwise identical")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
