// Named pipelines and their option strings.
// RUN: not ops-opt %s -ops-jit-cpu="bogus=1" 2>&1 | FileCheck %s --check-prefix=BOGUS
// RUN: not ops-opt %s -ops-jit-cpu="omp=maybe" 2>&1 | FileCheck %s --check-prefix=BADBOOL
// RUN: not ops-opt %s -ops-jit-cpu="specialize=everything" 2>&1 | FileCheck %s --check-prefix=BADMODE
// RUN: not ops-opt %s -ops-jit-plan="placement=middle" 2>&1 | FileCheck %s --check-prefix=PLACEMENT
// RUN: not ops-opt %s -ops-jit-frontend="anything=1" 2>&1 | FileCheck %s --check-prefix=NOOPTS
// RUN: ops-opt %s -ops-jit-plan="max-loops=2 reorder=false" | FileCheck %s
// RUN: ops-opt %s -ops-jit-plan="max-loops=2,reorder=false" | FileCheck %s

// BOGUS: unknown option 'bogus' (known: omp, specialize, soa, fma)
// BADBOOL: option 'omp' expects true or false, got 'maybe'
// BADMODE: option 'specialize' must be none, ranges or shapes, got 'everything'
// PLACEMENT: option 'placement' must be earliest or latest, got 'middle'
// NOOPTS: unknown option 'anything' (there are none)
// CHECK: ops.chain
ops.chain @empty() {
  ops.return
}
