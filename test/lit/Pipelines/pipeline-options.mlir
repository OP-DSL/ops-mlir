// Named pipelines and their option strings.
// RUN: not ops-opt %s -ops-jit-cpu="bogus=1" 2>&1 | FileCheck %s --check-prefix=BOGUS
// RUN: not ops-opt %s -ops-jit-cpu="omp=maybe" 2>&1 | FileCheck %s --check-prefix=BADBOOL
// RUN: not ops-opt %s -ops-jit-cpu="specialize=everything" 2>&1 | FileCheck %s --check-prefix=BADMODE
// RUN: not ops-opt %s -ops-jit-cpu="loops=poly" 2>&1 | FileCheck %s --check-prefix=BADLOOPS
// RUN: not ops-opt %s -ops-jit-cpu="affine-passes=func.func(affine-loop-tile)" 2>&1 | FileCheck %s --check-prefix=NEEDAFFINE
// RUN: not ops-opt %s -ops-jit-cpu="loops=affine affine-passes=no-such-pass" 2>&1 | FileCheck %s --check-prefix=NOPASS
// RUN: not ops-opt %s -ops-plan="placement=middle" 2>&1 | FileCheck %s --check-prefix=PLACEMENT
// RUN: not ops-opt %s -ops-plan="guarded=true" 2>&1 | FileCheck %s --check-prefix=GUARDED
// RUN: not ops-opt %s -ops-frontend="anything=1" 2>&1 | FileCheck %s --check-prefix=NOOPTS
// RUN: ops-opt %s -ops-plan="max-loops=2 reorder=false placement=latest box-ratio=2.0" | FileCheck %s

// BOGUS: unknown option 'bogus' (known: omp, specialize, soa, fma, loops, affine-passes)
// BADBOOL: option 'omp' expects true or false, got 'maybe'
// BADMODE: option 'specialize' must be none, ranges or shapes, got 'everything'
// BADLOOPS: option 'loops' must be scf or affine, got 'poly'
// NEEDAFFINE: option 'affine-passes' needs loops=affine
// NOPASS: option 'affine-passes':
// PLACEMENT: ops-fuse-point-local: unknown placement 'middle' (earliest or latest)
// GUARDED: ops-fuse-point-local: guarded fusion is not implemented yet
// NOOPTS: no such option anything
// CHECK: ops.chain
ops.chain @empty() {
  ops.return
}
