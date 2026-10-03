# CloverLeaf 2D and 3D through ops-mlir

> This is the Simplified Technical English (ASD-STE100) version of [cloverleaf.md](../cloverleaf.md). Code, listings and numbers are the same as in the original.

[CloverLeaf](https://github.com/UK-MAC/CloverLeaf) is a compressible-Euler mini-app. The
OPS repository has 2D and 3D ports (`apps/c/CloverLeaf`, `apps/c/CloverLeaf_3D`).
These ports are the first real application that runs through ops-mlir. They have dozens of user kernels
and about 100 loops for each time step. They also have int and double dats, 1-D coordinate arrays inside
2-D and 3-D loops, data-dependent upwinding, global flags, and constants that change every step. They
also have reductions.

The project uses the application sources **unmodified, straight from the OPS checkout**. They are GPL-3
and this project is MIT, so the project copies nothing. `apps/c/cloverleaf_{2d,3d}` contain only a build
file, a shim header and a small `main()` wrapper.

* [1. How the parts connect](#1-how-the-parts-connect)
* [2. What the translator accepts](#2-what-the-translator-accepts)
* [3. Correctness: how the project checks it](#3-correctness-how-the-project-checks-it)
* [4. Results](#4-results)
* [5. Limitations](#5-limitations)

## 1. How the parts connect

```
 CloverLeaf sources (unmodified)                 apps/c/cloverleaf_2d|3d
 ───────────────────────────────                 ───────────────────────────────
 #include "ops_seq_v2.h" (2D) / "ops_seq.h" (3D)  shim/ops_seq*.h   <- found first on the include path
        │                                              │ 1. #define ops_par_loop ops_par_loop_stock
        │                                              │    #include_next the real header   (stock OPS)
        │                                              │ 2. include ops/OPSWrapper.h        (queueing ops_par_loop)
        │                                              │ 3. ops_decl_const also registers the constant
        ▼
 ops_par_loop(kernel, "label", block, dims, range, args...)
        │
        ▼  OPSWrapper.h
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ LoopDesc { kernel address, label, range, ArgDesc[], fallback closure }       │
 │   • kernel identity  = function address, resolved with dladdr (-rdynamic)    │
 │   • global elem_kind = deduced from the kernel signature (int vs float)      │
 │   • fallback         = copy of the ops_args that runs the STOCK ops_par_loop │
 └──────────────────────────────────────────────────────────────────────────────┘
        │  enqueue:  probe the kernel once (translator), and if it translates,
        │            append the registered constants it reads as synthetic read-only globals
        ▼
 JITEngine queue ──flush──► for each run of consecutive loops:
        │
        ├─ translatable ──► FusionPlanner ─► ops.par_loop IR ─► xDSL (stencil) ─► MLIR ─► seq | OpenMP | CUDA
        │
        └─ not translatable (none in CloverLeaf any more; any kernel the translator rejects)
                          ──► dats made current on the host, then the stock loop runs, in queue order
```

**Why a fallback.** Every loop has two implementations: the JIT implementation and the stock
implementation. If the translator cannot translate a user kernel yet, the loop runs through the stock
implementation at its place in the queue. So the program is always complete and correct.

The report says how much of the program is JIT-compiled. `OPS_MLIR_STATS=1` prints
`coverage: N loops JIT-compiled, M through the stock fallback`. `OPS_MLIR_EXPLAIN=1` prints `[jit]` or
`[host]` with the reason for each user kernel. CloverLeaf 2D and 3D do not need the fallback now,
because the JIT compiles every one of their loops (section 2.1).

**Kernel identity.** CloverLeaf gives the label `"update_halo_kernel1"` to a dozen different user
kernels, so the label cannot name the user kernel. The wrapper takes the address of the user kernel
(the first argument of the loop). The runtime resolves the address with `dladdr` to the function name.
The translator uses the same name to find the function in the kernel headers.
(The link of the executable uses `-rdynamic`.)

**Constants.** User kernels read application globals (`dt`, `g_small`, `field.x_max`, ...). The shim
registers every `ops_decl_const` global with the runtime. The translator turns each registered global
that a user kernel reads into a trailing scalar parameter. The runtime takes a snapshot of its value
when the loop enters the queue. Then it passes the value like any read-only global.

The project must not bake the value into the module. This is wrong for `dt`, which changes every step. It also makes the
key in the module cache depend on a value.

**Element kinds.** `ops_arg_gbl(fields, 15, "int", OPS_READ)` records only `sizeof(int) == 4` in the
`ops_arg`. An `int` flag array and a `float` array look the same. So the wrapper reads the element
type from the parameter of the user kernel (`const int *fields`). It records the type in the loop
(`ArgAttr.elem_kind`). Each element of an array global becomes one scalar parameter.

## 2. What the translator accepts

`lib/runtime/AccessorKernel.cpp` translates the Clang AST of a user kernel into an MLIR `func` of
scalars. The signature follows from the arguments of the loop. It has one input for each (read dat,
stencil point). Then it has the elements of the read-only globals, then the registered constants.
The output is the returned value (one written dat) or a trailing `memref` (more than one written dat).

The translator knows two styles of user kernel:

| style | example |
|---|---|
| accessor (2D port) | `void k(const ACC<double> &a, ACC<double> &b, const int *flags) { b(0,0) = a(1,0) * ... }` |
| pointer (3D port) | `void k(const double *a, double *b, const int *flags) { b[OPS_ACC1(0,0,0)] = a[OPS_ACC0(1,0,0)] * ... }` |

The translator has its own definition of `OPS_ACCn(x,y,z)`: `__ops_acc(n, x, y, z)`. So the index
carries the argument that it belongs to, and the translator rejects a mismatch.

The translator supports these things:

* local variables and arithmetic in `float`, `double` and integer types
* `if`/`else`
* constant `for` loops (the translator unrolls them)
* `?:`
* the usual math functions
* `MIN`/`MAX`/`SIGN` with the definitions of OPS
* enumerators
* `fields[CONST]` on array globals
* members of struct constants (`field.x_max`)
* **data-dependent offsets**

These notes explain the less obvious things:

* **Conditionals become selects.** The two branches compute only on values that the user kernel already
  loaded. So the translator runs both branches unconditionally and merges them with `arith.select`.

  There are no regions inside the body of the user kernel. The translator inlines the body into a
  `memref.alloca_scope`, which must stay a single block. The translator rejects an integer division
  inside a conditional, because it must not run speculatively. A write-only output that both branches
  assign needs no value from before the branches.
* **Data-dependent offsets** (`density1(donor, 0)` in `advec_cell_kernel3`, where the sign of a flux
  chooses `donor`) are always one of the declared points of the stencil. The translator reads all the
  points. It selects the point with the matching offset.
* **Strided stencils** (`S2D_00_STRID2D_X`: a 1-D coordinate array that a 2-D loop uses). The dat
  becomes a *lower-rank field*. Every access is a `stencil.access` with an `offset_mapping` onto the
  axes of the loop. The stencil-to-memref lowering already supports this. Section 2.1 describes writes.
* **Unresolved names.** The translator parses the kernel header with the includes that the translation
  unit of the application had before it (`set_kernel_preamble`), and with stubs for OPS types. If the
  body still has an unresolved name, the translator rejects the user kernel. It does not guess.

### 2.1 The last three kinds of loop: reductions, 1-D arrays that the loop writes, arrays of structs

These loops stayed on the stock implementation in the first version (166 of 11 751 loops in 2D).
On the GPU, they forced the program to copy the reduced dats to the host in each step. The JIT
compiles them now.

* **Reductions** (`calc_dt_kernel_min`, `calc_dt_kernel_get`, `field_summary_kernel`). The loop does
  not accumulate a reduction argument in place. The translated user kernel starts each reduction
  element at the identity of the operation (0, +max, −max). It returns the *contribution* of this
  point as an extra output. The generated kernel stores the contribution into a scratch buffer over
  the box of the group. The runtime first fills the scratch buffer with the identity. So the points
  that a guarded (fused) member does not visit contribute nothing.

  After the loop, the runtime folds
  the scratch buffer to one value with a small native helper. Then it combines the value with the OPS
  reduction handle (`+`, `min`, `max`).

  On the host, the helper adds a fixed number of chunks in order. On the GPU, the helper uses two
  PTX GPU kernels. The `fill` GPU kernel sets the identity.

  The `red` GPU kernel runs 2 times. The first
  run gives each thread a grid-stride slice and does a block fold in shared memory. The second run is a
  single block over the results of the blocks. The helper loads the PTX GPU kernels through the driver
  API and launches them on the same stream.

  The fold order depends on the box and not on the thread count, so the results are deterministic.
  But the order is *not* the sequential order. A sum differs from the stock result in the last bits.
  (The 3D QA line reads 5.7e-14 % where stock has 2.8e-14 %. Both are far under the pass level of
  1e-3 %. MIN and MAX are exact.)

  The translator recognizes `*r = *r + e` as an accumulation. Any other assignment to an INC argument (`*r = e`) is equivalent only for a loop over one point.
  So such a loop with more points stays on the host (CloverLeaf has none).

  The planner still treats a loop with a reduction as a fusion barrier.
* **Loops that write 1-D arrays** (`initialise_chunk_kernel_x/y/xx/yy/cellx/celly`: `vertexx` has an
  index along x only, but the loop is 2-D). The stock loop writes each element 1 time for each row,
  with the same value each time. The generated kernel iterates over the axes that the dats have. It
  visits each element 1 time.

  This gives the same result only if nothing in the loop depends on the
  dropped axes. The runtime checks this.

  Every dat of the loop has an index along the same axes. The
  loop accesses the dats only with `WRITE` or `READ`. The loop has no reduction. The translator reports
  that the user kernel never reads `idx[k]` of a dropped axis. The planner does not fuse loops whose
  dats span different axes.
* **Registered arrays of structs** (`generate_chunk_kernel` reads `states[i].energy` inside
  `for (i = 1; i < number_of_states; ...)`). The translator unrolls the loop with the registered
  `number_of_states` (a translation-time constant). Each `states[i].field` with constant `i` becomes a
  scalar parameter. The runtime reads its value from the registered array when the loop enters the queue.

  The translator knows the elements only when it emits the body. So the translator runs 2 times. The
  first pass finds the elements. The second pass has them in its signature. A constant that the
  translator folds into the code in this way is part of the key of the compiled module in the cache.
  A change of the constant makes the translator translate the user kernel again (tested in
  `tests/e2e/accessor_cases.cpp`).

Single-point loops (`calc_dt_kernel_get` runs over one cell) need one more thing on the GPU.
Canonicalisation removes a `scf.parallel` that has one iteration. So no loop remained to map to a GPU
kernel, and the body ran on the host against device pointers (a segmentation fault). The mapping pass
now wraps such a function in a launch with one thread.

## 3. Correctness: how the project checks it

1. **CloverLeaf's own QA.** Each deck has an expected kinetic energy. The program prints
   `Test problem N is within X % of the expected solution`. The test passes when the value is below
   `1e-3 %`. Every configuration below passes. The printed value is the same as the stock value only where
   section 4.1 says so. The runtime now folds the reductions in a different order, so the last digits can differ.
2. **A differential verifier** (`OPS_MLIR_VERIFY=1`). The verifier runs each loop that the JIT compiles
   2 times from the same state. One run is JIT-compiled and one run goes through the stock
   implementation.

   The verifier compares *all* dats of the OPS instance bit for bit. (An out-of-bounds
   write corrupts a dat that the loop does not even name.) It also compares the reduction handles: sums
   to 1e-12 relative, min/max exactly. The verifier keeps the stock result, so the run continues
   unharmed. The report counts the mismatches for each user kernel
   (`ops-mlir verify: <kernel>: 0 of 75 runs differ`).

   The verifier found two real bugs. One bug was a strided stencil that the lowering mis-indexed.
   The other bug was an `if` whose body the translator parsed without its enumerators, so the body
   silently read them as 0.

A third item belongs here too. The QA value changed with the size of the environment. This became
clear when the verifier was built.

The cause was a use-after-free in the fallback closure. The closure
did not keep the copies of the read-only globals alive. So the bug was in this project and not in OPS
or CloverLeaf. After the fix, the result does not depend on the environment. `OPS_MLIR_HOST=all`
(everything through the stock path) gives exactly the same result as the stock executable.

## 4. Results

Everything below ran on `renyi` (2× A100-40GB with one in use, 16 CPU threads) with the commits listed in
`git log`. The raw data is in `docs/data/clover_a100_qa.txt` and
`docs/data/clover_fusion_a100_{2d,3d}.json`.
The decks are `clover.in_default` (QA problem 1, tiny grid) and `clover_bm_short.in` (QA problem 2).
`clover_bm_short.in` has 960² cells in 2D, 96³ cells in 3D, and 87 steps.

### 4.1 Coverage and correctness

The JIT compiles every loop of both applications (`docs/data/clover_a100_qa_allgpu.txt`). The first
version of this section had 98.6–99.6 % coverage. This version replaces it.

| | loops | JIT-compiled | stock fallback | user kernels checked bit for bit (`VERIFY`, seq/OpenMP) |
|---|---:|---:|---:|---:|
| 2D default | 11 751 | 11 751 | 0 | 82, none differ |
| 2D `bm_short` | 13 625 | 13 625 | 0 | – |
| 3D default | 45 339 | 45 339 | 0 | 140, none differ |
| 3D `bm_short` | 52 577 | 52 577 | 0 | – |

The QA of CloverLeaf for every configuration (A100 for CUDA):

| deck | stock | ops-mlir seq | OpenMP (16) | CUDA (A100) |
|---|---|---|---|---|
| 2D default | 2.842e-14 % PASSED | identical | identical | identical |
| 2D `bm_short` | 8.527e-14 % PASSED | 1.164e-11 % PASSED | 1.164e-11 % | 1.168e-11 % |
| 3D default | 2.842e-14 % PASSED | 5.684e-14 % | 5.684e-14 % | 4.263e-14 % |
| 3D `bm_short` | 8.527e-14 % PASSED | 8.697e-12 % | 8.697e-12 % | 8.697e-12 % |

The QA values are no longer all identical to the stock values. The runtime now folds reductions in a
fixed parallel order and not in the sequential order. The differences are in the last digits of a
sum over up to 10⁶ cells (a relative 1e-11 on the kinetic-energy check). This is 8 orders of magnitude
below the pass level of 1e-3 %. Every per-loop comparison of the dats themselves is still bit-exact.
(Before this change, the QA lines were identical because those loops ran through the stock code.)

### 4.2 Fusion

The variants are (`apps/c/cloverleaf_2d/eval_fusion.py`):

* `off` (`OPS_MLIR_FUSION=0`)
* `consecutive` (only adjacent loops fuse)
* `dag` (the planner for the producer/consumer DAG, the default)
* `dag-latest` (late placement)
* `dag-max32` (up to 32 loops for each generated kernel, the default is 8)

*Steady state* is `execute + host-fallback` seconds. So it excludes the compile time and the cost to put
each loop in the queue. Fusion changes neither of them. Every row reproduced the stock QA value.

The measurements in 4.2 and 4.3 come from the first version. In that version the reductions and the one-time
setup loops still ran as stock loops (`host-fallback`). When every loop runs on the GPU, the times are lower.
See the update at the top of [cloverleaf_large.md](cloverleaf_large.md).

**2D, 960², 87 steps** (13 434 loops)

| backend | variant | generated kernel launches | steady state | vs `off` |
|---|---|---:|---:|---:|
| seq | off | 13 434 | 15.1 s | |
| seq | consecutive | 12 999 (−3 %) | 17.9 s | +18 % |
| seq | dag | 8 902 (−34 %) | 17.8 s | +18 % |
| OpenMP | off | 13 434 | 12.6 s | |
| OpenMP | consecutive | 12 999 | 13.0 s | +3 % |
| OpenMP | dag | 8 902 (−34 %) | 11.4 s | −9 % |
| OpenMP | dag-max32 | 8 902 | 11.2 s | −11 % |
| CUDA A100 | off | 13 434 | 4.76 s | |
| CUDA A100 | consecutive | 12 999 | 4.70 s | −1 % |
| CUDA A100 | dag | 8 902 (−34 %) | 4.35 s | −9 % |

**3D, 96³, 87 steps** (52 383 loops)

| backend | variant | generated kernel launches | steady state | vs `off` |
|---|---|---:|---:|---:|
| OpenMP | off | 52 383 | 19.3 s | |
| OpenMP | consecutive | 52 035 (−1 %) | 19.3 s | 0 % |
| OpenMP | dag | 28 852 (−45 %) | 14.9 s | −23 % |
| OpenMP | dag-latest | 28 852 | 14.6 s | −24 % |
| CUDA A100 | off | 52 383 | 10.3 s | |
| CUDA A100 | consecutive | 52 035 | 10.2 s | −1 % |
| CUDA A100 | dag | 28 852 (−45 %) | 9.5 s | −7 % |

The numbers show these things:

* **The planner must reorder loops to fuse the loops of CloverLeaf.** Fusion of adjacent loops only
  removes 1–3 % of the launches, because the loops of CloverLeaf alternate between producers and
  unrelated consumers. The DAG planner moves a loop across the loops that it commutes with. It removes
  34 % (2D) and 45 % (3D).
* **The gain in time is smaller than the gain in launches, and it is not uniform.** 3D on OpenMP gains
  the most (−23 %). On the A100 the gain is 7–9 %. In 2D on one thread, fusion is *slower* (+18 %).
  The generated kernels have more live values. I did not investigate why the sequential code is
  slower, so the cause is unconfirmed. It is a measured regression and not a noise effect
  (3 variants, same result).
* **Other things than the GPU kernels dominate the A100 numbers on the 960² / 96³ decks.** The
  reductions of the first version in the host fallback (`calc_dt_kernel_min`, `field_summary_kernel`) took more time than all
  GPU kernels. A whole-run "kernel seconds" figure includes a one-off cost of ≈ 0.8 s the first time
  that each compiled module launches. (An earlier version of this text blamed ≈ 0.44 ms of overhead for
  each launch. That was wrong. The steady overhead for each launch is 15–30 µs, as measured.)
  See [cloverleaf_large.md](cloverleaf_large.md). It separates these costs and uses the larger decks.
* Group size (8 → 32) and placement (earliest/latest) make no measurable difference. The dependences
  bound the groups here, and not the cap.

The larger decks (2D 3840², 3D 256³) are in [cloverleaf_large.md](cloverleaf_large.md). That document
has Nsight Compute, register pressure and a measured breakdown of where the step time goes. That study
found that a concurrent job on the shared node contaminated some of the timings above. It redid the A100
sweeps with repetitions. Its numbers replace the CUDA figures in this section.

### 4.3 Comparison with the stock implementation

This is for orientation only. The stock sequential `ops_par_loop` is the development path of OPS (a
generic loop that calls the user kernel through accessor objects). It is not a code-generated backend of
OPS, so it is a weak baseline.

| deck | stock (1 thread) | ops-mlir steady state, seq | OpenMP (16) | CUDA A100 |
|---|---:|---:|---:|---:|
| 2D `bm_short` | 45.8 s | 15.1 s | 12.6 s | 4.8 s |
| 3D `bm_short` | 73.6 s | – | 19.3 s | 10.3 s |

The ops-mlir values are for `off`, that is, without fusion, to isolate the translation. They exclude the
one-off compile time below.

### 4.4 One-off costs

The project pays for compilation 1 time for each different queue of loops, and CloverLeaf has 7 queues
for each deck. On renyi (final build, from the fusion runs) this takes 26–43 s in 2D and 41–86 s in 3D.
CUDA is the slow one.

The first correctness runs on the cluster spent 100–110 s (2D) and 200 s (3D) on
compilation. These runs were before the overhead fixes of this work (plan cache, memoised xDSL type
conversion, symbol cache). I measured locally on 2D `bm_short`, with one thread. Those fixes reduced the
whole run from 94 s to 39 s (xDSL lowering 59 s → 9 s, enqueue 10 s → 2.6 s). On 3D the planner alone
cost 110 s. It now costs about 0.1 s.

The fixed compile cost still dominates the end-to-end wall time of these short decks. In a real
CloverLeaf run of thousands of steps, the fixed compile cost does not dominate.

## 5. Limitations

* **The runtime folds reductions in a different order than the sequential loop adds them.** The result
  is deterministic, but for sums it is not bitwise equal to stock. Min and max are exact. See 2.1 and
  the QA table.
* **Reduction loops are fusion barriers**, and the fold runs 1 time for each loop.
* **`*r = e` on an INC reduction** over more than one point stays on the stock path (none in
  CloverLeaf).
* **Data-dependent offsets read every point of the stencil** and select. This is correct for any offset
  inside the declared stencil. But if a user kernel indexes outside the stencil, the translator mistranslates it
  and does not reject it. (The stock code reads whatever is in memory there, so no valid
  application does this.)
* **The translator bakes registered integer constants that the loop uses as compile-time loop bounds.**
  The translator needs them at translation time to unroll. The value is part of the module key, and a
  change of the value makes the translator translate the user kernel again. But this works only if the
  value changes between flushes, and not between the enqueue of a loop and its flush.
* **The GPU launch path is the bottleneck** (see 4.2). So the absolute GPU times here say little about
  the fusion gain that a leaner runtime can reach.
* **The sequential regression with fusion** (+18 % in 2D) has no explanation.
* The project tested only the sequential, OpenMP and CUDA backends, in a single process. It did not try
  MPI.
