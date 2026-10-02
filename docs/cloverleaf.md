# CloverLeaf 2D and 3D through ops-mlir

[CloverLeaf](https://github.com/UK-MAC/CloverLeaf) is a compressible-Euler mini-app; the
OPS repository carries 2D and 3D ports (`apps/c/CloverLeaf`, `apps/c/CloverLeaf_3D`).
These ports are the first real application run through ops-mlir: dozens of kernels, ~100
loops per time step, int and double dats, 1-D coordinate arrays inside 2-D and 3-D loops,
data-dependent upwinding, global flags, constants that change every step, and reductions.

The application sources are **used unmodified, straight from the OPS checkout** (they are GPL-3;
this project is MIT, so nothing is copied). `apps/c/cloverleaf_{2d,3d}` only contain a build
file, a shim header and a small `main()` wrapper.

* [1. How it is wired up](#1-how-it-is-wired-up)
* [2. What the translator accepts](#2-what-the-translator-accepts)
* [3. Correctness: how it is checked](#3-correctness-how-it-is-checked)
* [4. Results](#4-results)
* [5. Limitations](#5-limitations)

## 1. How it is wired up

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

**Why a fallback.** Every loop has two implementations: the JIT's and the stock one. A kernel the
translator cannot handle yet runs through the stock implementation, at its place in the queue, so the
program is always complete and correct, and the report says exactly how much of it is JIT-compiled
(`OPS_MLIR_STATS=1` prints `coverage: N loops JIT-compiled, M through the stock fallback`).
`OPS_MLIR_EXPLAIN=1` prints, for each kernel, either `[jit]` or `[host]` with the reason. CloverLeaf 2D and 3D
no longer need it: every one of their loops is JIT-compiled (section 2.1).

**Kernel identity.** CloverLeaf labels a dozen different kernels `"update_halo_kernel1"`, so the label
cannot name the kernel. The wrapper takes the kernel's address (the loop's first argument) and the
runtime resolves it with `dladdr` to the function name, which is also the function the translator looks
up in the kernel headers. (The executable is linked with `-rdynamic`.)

**Constants.** Kernels read application globals (`dt`, `g_small`, `field.x_max`, ...). The shim registers
every `ops_decl_const` global with the runtime. The translator turns each one a kernel reads into a
trailing scalar parameter, and the runtime snapshots its value when the loop is *enqueued* and passes it
like any read-only global. Baking the value into the module would be wrong for `dt`, which changes every
step, and would also make the module cache key depend on a value.

**Element kinds.** `ops_arg_gbl(fields, 15, "int", OPS_READ)` records only `sizeof(int) == 4` in the
`ops_arg`; an `int` flag array and a `float` array look the same. The wrapper therefore reads the
element type from the kernel's own parameter (`const int *fields`) and records it in the loop
(`ArgAttr.elem_kind`). Array globals become one scalar parameter per element.

## 2. What the translator accepts

`lib/runtime/AccessorKernel.cpp` translates a kernel's Clang AST into an MLIR `func` of
scalars. The signature follows from the loop's arguments: one input per (read dat, stencil point), then
the elements of the read-only globals, then the registered constants; the output is the returned value
(one written dat) or a trailing `memref` (several).

Two kernel styles are recognised:

| style | example |
|---|---|
| accessor (2D port) | `void k(const ACC<double> &a, ACC<double> &b, const int *flags) { b(0,0) = a(1,0) * ... }` |
| pointer (3D port) | `void k(const double *a, double *b, const int *flags) { b[OPS_ACC1(0,0,0)] = a[OPS_ACC0(1,0,0)] * ... }` |

The translator's own definition of `OPS_ACCn(x,y,z)` is `__ops_acc(n, x, y, z)`, so the index carries
the argument it belongs to and a mismatch is rejected.

Supported: local variables and arithmetic in `float`, `double` and integer types, `if`/`else`, constant
`for` loops (unrolled), `?:`, the usual math functions, `MIN`/`MAX`/`SIGN` with OPS's definitions,
enumerators, `fields[CONST]` on array globals, members of struct constants (`field.x_max`), and
**data-dependent offsets**.

Notes on the less obvious ones:

* **Conditionals become selects.** Both branches only compute on values that are already loaded, so the
  translator runs them unconditionally and merges with `arith.select` — no regions inside the kernel
  body, which is inlined into a `memref.alloca_scope` that must stay a single block. Integer division
  inside a conditional is rejected (it must not run speculatively). A write-only output assigned in
  both branches needs no prior value.
* **Data-dependent offsets** (`density1(donor, 0)` with `donor` chosen from the sign of a flux in
  `advec_cell_kernel3`) are always one of the stencil's declared points. All the points are read and the
  one whose offset matches is selected.
* **Strided stencils** (`S2D_00_STRID2D_X`: a 1-D coordinate array used inside a 2-D loop). The dat
  becomes a *lower-rank field* and every access is a `stencil.access` with an `offset_mapping` onto the
  loop's axes, which the stencil-to-memref lowering already supports. Writes are covered in 2.1.
* **Unresolved names.** The kernel header is parsed with the includes the application's own translation
  unit had in front of it (`set_kernel_preamble`) plus stubs for OPS types. If the body still contains
  an unresolved name the kernel is rejected rather than guessed at.

### 2.1 The last three kinds of loop: reductions, 1-D arrays written, struct arrays

These stayed on the stock implementation in the first version (166 of 11 751 loops in 2D, but on the GPU
they forced the reduced dats to be copied to the host each step). They are compiled now.

* **Reductions** (`calc_dt_kernel_min`, `calc_dt_kernel_get`, `field_summary_kernel`). A reduction
  argument is not accumulated in place. The translated kernel starts every reduction element at the
  operation's identity (0, +max, −max) and returns this point's *contribution* as an extra output,
  which the generated function stores into a scratch field over the group's iteration box. The
  runtime fills that field with the identity first, so points a guarded (fused) member does not visit
  contribute nothing, and afterwards folds it to one value with a small native helper (host: a fixed
  number of chunks added in order; GPU: a PTX `fill` kernel for the identity and a PTX `red` kernel that runs twice, once
  with a grid-stride slice per thread and a block fold in shared memory, once as a single block over the
  per-block results; loaded through the driver API and launched on the same stream) and combines that with the
  OPS reduction handle (`+`, `min`, `max`). The fold order depends on the box and not on the thread
  count, so results are deterministic, but it is *not* the sequential order: a sum differs from the
  stock result in the last bits (the 3D QA line reads 5.7e-14 % where stock has 2.8e-14 %; both far under
  the 1e-3 % pass level; MIN and MAX are exact). `*r = *r + e` is recognised as an accumulation;
  any other assignment to an INC argument (`*r = e`) is only equivalent for a loop over one point, so such
  a loop with more points stays on the host (CloverLeaf has none).
  The planner still treats a loop with a reduction as a fusion barrier.
* **Loops that write 1-D arrays** (`initialise_chunk_kernel_x/y/xx/yy/cellx/celly`: `vertexx` is
  indexed along x only but the loop is 2-D). The stock loop writes each element once per row, the same value
  each time. The generated function iterates over the axes the dats have and visits every element once.
  That is only the same if nothing in the loop depends on the dropped axes, which the runtime checks: every dat
  of the loop is indexed along the same axes, they are only `WRITE`n or `READ`, no reduction, and the
  translator reports that the kernel never reads `idx[k]` of a dropped axis. The planner does not fuse loops
  whose dats span different axes.
* **Registered arrays of structs** (`generate_chunk_kernel` reads `states[i].energy` inside
  `for (i = 1; i < number_of_states; ...)`). The loop is unrolled with the registered `number_of_states`
  (a translation-time constant), and each `states[i].field` with constant `i` becomes a scalar parameter whose
  value is read from the registered array when the loop is enqueued. The elements are only known while the
  body is emitted, so the translator runs twice: the first pass finds them, the second has them in its signature.
  A constant folded into the code like this is part of the compiled module's cache key, and changing it
  re-translates the kernel (tested in `tests/e2e/accessor_cases.cpp`).

Single-point loops (`calc_dt_kernel_get` runs over one cell) need one more thing on the GPU: canonicalisation
removes a one-iteration `scf.parallel`, so nothing was left to map to a kernel and the body ran on the host
against device pointers (a segmentation fault). The mapping pass now wraps such a function in a one-thread launch.

## 3. Correctness: how it is checked

1. **CloverLeaf's own QA.** Each deck has an expected kinetic energy; the program prints
   `Test problem N is within X % of the expected solution` and passes below `1e-3 %`.
   Every configuration below reproduces the *stock* value of that line.
2. **A differential verifier** (`OPS_MLIR_VERIFY=1`). Every loop that the JIT compiles is run twice
   from the same state — once JIT-compiled, once through the stock implementation — and *all* dats of
   the OPS instance are compared bit for bit (an out-of-bounds write corrupts a dat the loop does not
   even name). Reduction handles are compared too (sums to 1e-12 relative, min/max exactly). The stock
   result is kept, so the run continues unharmed, and the report counts the
   mismatches per kernel (`ops-mlir verify: <kernel>: 0 of 75 runs differ`).
   It found two real bugs: a strided stencil that the lowering mis-indexed, and an `if`
   whose body was parsed without its enumerators and silently read as 0.

A third item belongs here too: while building the verifier the QA value was found to change with the
size of the environment. The cause was a use-after-free in the fallback closure (it did not keep the
copies of the read-only globals alive), i.e. in this project and not in OPS or CloverLeaf. With that fixed
the result is independent of the environment, and `OPS_MLIR_HOST=all` (everything through the stock path)
equals the stock executable exactly.

## 4. Results

Everything below ran on `renyi` (2× A100-40GB, one used; 16 CPU threads) with the commits listed in
`git log`; raw data is in `docs/data/clover_a100_qa.txt` and `docs/data/clover_fusion_a100_{2d,3d}.json`.
Decks: `clover.in_default` (QA problem 1, tiny grid), `clover_bm_short.in` (QA problem 2; 960² cells
in 2D, 96³ in 3D, 87 steps).

### 4.1 Coverage and correctness

Every loop of both applications is JIT-compiled (`docs/data/clover_a100_qa_allgpu.txt`; the first version of this
section, with 98.6–99.6 % coverage, is superseded):

| | loops | JIT-compiled | stock fallback | kernels checked bitwise (`VERIFY`, seq/OpenMP) |
|---|---:|---:|---:|---:|
| 2D default | 11 751 | 11 751 | 0 | 82, none differ |
| 2D `bm_short` | 13 625 | 13 625 | 0 | – |
| 3D default | 45 339 | 45 339 | 0 | 140, none differ |
| 3D `bm_short` | 52 577 | 52 577 | 0 | – |

CloverLeaf's own QA, every configuration (A100 for CUDA):

| deck | stock | ops-mlir seq | OpenMP (16) | CUDA (A100) |
|---|---|---|---|---|
| 2D default | 2.842e-14 % PASSED | identical | identical | identical |
| 2D `bm_short` | 8.527e-14 % PASSED | 1.164e-11 % PASSED | 1.164e-11 % | 1.168e-11 % |
| 3D default | 2.842e-14 % PASSED | 5.684e-14 % | 5.684e-14 % | 4.263e-14 % |
| 3D `bm_short` | 8.527e-14 % PASSED | 8.697e-12 % | 8.697e-12 % | 8.697e-12 % |

The QA values are no longer all identical to stock's, because reductions are now folded in a fixed
parallel order instead of the sequential one. The differences are in the last digits of a sum over up to
10⁶ cells (a relative 1e-11 on the kinetic-energy check), 8 orders of magnitude below the 1e-3 % pass level;
every per-loop comparison of the dats themselves is still bit-exact. (Before this change the QA lines were
identical because those loops ran through the stock code.)

### 4.2 Fusion

Variants (`apps/c/cloverleaf_2d/eval_fusion.py`): `off` (`OPS_MLIR_FUSION=0`), `consecutive`
(only adjacent loops fuse), `dag` (the producer/consumer DAG planner, the default), `dag-latest`
(late placement), `dag-max32` (up to 32 loops per kernel; default 8). *Steady state* is
`execute + host-fallback` seconds, so it excludes compilation and the per-loop enqueue cost, neither of
which fusion changes. Every row reproduced the stock QA value.

**2D, 960², 87 steps** (13 434 loops)

| backend | variant | kernel launches | steady state | vs `off` |
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

| backend | variant | kernel launches | steady state | vs `off` |
|---|---|---:|---:|---:|
| OpenMP | off | 52 383 | 19.3 s | |
| OpenMP | consecutive | 52 035 (−1 %) | 19.3 s | 0 % |
| OpenMP | dag | 28 852 (−45 %) | 14.9 s | −23 % |
| OpenMP | dag-latest | 28 852 | 14.6 s | −24 % |
| CUDA A100 | off | 52 383 | 10.3 s | |
| CUDA A100 | consecutive | 52 035 | 10.2 s | −1 % |
| CUDA A100 | dag | 28 852 (−45 %) | 9.5 s | −7 % |

What the numbers say:

* **Reordering is what makes CloverLeaf fusable.** Adjacent-only fusion removes 1–3 % of the launches
  because CloverLeaf's loops alternate between producers and unrelated consumers; the DAG planner,
  which moves a loop across loops it commutes with, removes 34 % (2D) and 45 % (3D).
* **The time gain is smaller than the launch gain, and not uniform.** 3D on OpenMP gains the most
  (−23 %). On the A100 the gain is 7–9 %. In 2D on one thread fusion is *slower* (+18 %) — the fused
  kernels have more live values and I did not look into why the sequential code is slower, so the cause
  is unconfirmed; it is a measured regression, not a noise effect (three variants, same result).
* **The A100 numbers on the 960² / 96³ decks are dominated by things other than the kernels.** The
  host-fallback reductions (`calc_dt_kernel_min`, `field_summary_kernel`) take more time than all GPU kernels, and a whole-run
  "kernel seconds" figure includes a one-off cost of ≈ 0.8 s the first time each compiled module launches. (An earlier
  version of this text blamed ≈ 0.44 ms per-launch overhead; that was wrong. Measured per launch, the steady overhead is
  15–30 µs.) See [cloverleaf_large.md](cloverleaf_large.md), which separates these and uses the larger decks.
* Group size (8 → 32) and placement (earliest/latest) make no measurable difference: the dependences, not
  the cap, bound the groups here.

The larger decks (2D 3840², 3D 256³), with Nsight Compute, register pressure and a measured breakdown of where
the step time goes, are in [cloverleaf_large.md](cloverleaf_large.md). That study found the timings above to be
contaminated by a concurrent job on the shared node in places, and redid the A100 sweeps with repetitions; its
numbers supersede the CUDA figures in this section.

### 4.3 Against the stock implementation

For orientation only. The stock sequential `ops_par_loop` is OPS's development path (a generic loop that
calls the kernel through accessor objects), not OPS's code-generated backends, so it is a weak baseline.

| deck | stock (1 thread) | ops-mlir steady state, seq | OpenMP (16) | CUDA A100 |
|---|---:|---:|---:|---:|
| 2D `bm_short` | 45.8 s | 15.1 s | 12.6 s | 4.8 s |
| 3D `bm_short` | 73.6 s | – | 19.3 s | 10.3 s |

(`off`, i.e. without fusion, to isolate the translation. These exclude the one-off compile time below.)

### 4.4 One-off costs

Compilation is paid once per distinct queue of loops, and CloverLeaf has 7 of them per deck. On renyi
(final build, from the fusion runs) that is 26–43 s in 2D and 41–86 s in 3D, CUDA being the slow one.
The first correctness runs on the cluster, before the overhead fixes made while writing this (plan
cache, memoised xDSL type conversion, symbol cache), spent 100–110 s (2D) and 200 s (3D) compiling.
Measured locally on 2D `bm_short`, one thread, those fixes took the whole run from 94 s to 39 s
(xDSL lowering 59 s → 9 s, enqueue 10 s → 2.6 s); on 3D the planner alone had cost 110 s and now costs
about 0.1 s. The end-to-end wall time of these short decks is still dominated by the fixed compile cost;
a real CloverLeaf run of thousands of steps would not be.

## 5. Limitations

* **Reductions are folded in a different order than the sequential loop adds them** (deterministic, but
  not bitwise equal to stock for sums; min/max are exact), see 2.1 and the QA table.
* **Reduction loops are fusion barriers** and the fold runs once per loop.
* **`*r = e` on an INC reduction** over more than one point stays on the stock path (none in CloverLeaf).
* **Data-dependent offsets read every point of the stencil** and select; this is correct for any offset
  inside the declared stencil but a kernel that indexes outside it would be mistranslated rather than
  rejected. (The stock code reads whatever is in memory there, so no valid application does it.)
* **Registered integer constants used as compile-time loop bounds are baked** (they are needed at
  translation time to unroll). The value is part of the module key and a change re-translates the kernel,
  but only if it changes between flushes, not between a loop's enqueue and its flush.
* **The GPU launch path is the bottleneck** (see 4.2), so absolute GPU times here say little about the
  fusion gain achievable with a leaner runtime.
* **Sequential regression with fusion** (+18 % in 2D) is unexplained.
* Only the sequential, OpenMP and CUDA backends, in a single process, were exercised; MPI was not tried.
