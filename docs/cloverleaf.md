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
        └─ not translatable (reductions, one-time initialisation)
                          ──► dats made current on the host, then the stock loop runs, in queue order
```

**Why a fallback.** Every loop has two implementations: the JIT's and the stock one. A kernel the
translator cannot handle yet runs through the stock implementation, at its place in the queue, so the
program is always complete and correct, and the report says exactly how much of it is JIT-compiled
(`OPS_MLIR_STATS=1` prints `coverage: N loops JIT-compiled, M through the stock fallback`).
`OPS_MLIR_EXPLAIN=1` prints, for each kernel, either `[jit]` or `[host]` with the reason.

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
  loop's axes, which the stencil-to-memref lowering already supports. Such a dat can only be read.
* **Unresolved names.** The kernel header is parsed with the includes the application's own translation
  unit had in front of it (`set_kernel_preamble`) plus stubs for OPS types. If the body still contains
  an unresolved name the kernel is rejected rather than guessed at.

Not supported, so the loop stays on the stock implementation: reductions (`calc_dt`'s minimum,
`field_summary`), writes through a strided stencil (the one-time `initialise_chunk` loops), and kernels
that index a struct array through a pointer (`generate_chunk`).

## 3. Correctness: how it is checked

1. **CloverLeaf's own QA.** Each deck has an expected kinetic energy; the program prints
   `Test problem N is within X % of the expected solution` and passes below `1e-3 %`.
   Every configuration below reproduces the *stock* value of that line.
2. **A differential verifier** (`OPS_MLIR_VERIFY=1`). Every loop that the JIT compiles is run twice
   from the same state — once JIT-compiled, once through the stock implementation — and *all* dats of
   the OPS instance are compared bit for bit (an out-of-bounds write corrupts a dat the loop does not
   even name). The stock result is kept, so the run continues unharmed, and the report counts the
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

| | loops | JIT-compiled | stock fallback | kernels checked bitwise (`VERIFY`) |
|---|---:|---:|---:|---:|
| 2D default | 11 751 | 11 585 (98.6 %) | 166 | 72, none differ |
| 2D `bm_short` | 13 625 | 13 434 (98.6 %) | 191 | – |
| 3D default | 45 339 | 45 170 (99.6 %) | 169 | 127, none differ |
| 3D `bm_short` | 52 577 | 52 383 (99.6 %) | 194 | – |

The loops left on the stock path are the reductions (`calc_dt`'s minimum, `field_summary`, the
position look-ups) and the one-time grid setup (`initialise_chunk_*`, `generate_chunk`).

CloverLeaf's own QA, every configuration (`docs/data/clover_a100_qa.txt`):

| deck | stock | ops-mlir seq | OpenMP (16) | CUDA (A100) |
|---|---|---|---|---|
| 2D default | 2.842e-14 % PASSED | identical | identical | identical |
| 2D `bm_short` | 8.527e-14 % PASSED | identical | identical | identical |
| 3D default | 2.842e-14 % PASSED | identical | identical | identical |
| 3D `bm_short` | 8.527e-14 % PASSED | identical | identical | identical |

"Identical" means the printed 16-digit value is the same as the stock executable's, not just under the
`1e-3 %` threshold.

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
* **The A100 is nowhere near its memory bandwidth here.** 8 902 launches take ≈ 3.9 s, i.e. ≈ 0.44 ms per
  launch, whereas a 960² loop touching ten double dats moves very roughly 80 MB (≈ 0.05 ms at 1.5 TB/s). The launches are dominated by
  per-launch costs in the runtime (argument packing, a stream synchronisation after every group, and the
  host fallbacks' copies), so fusing fewer, larger kernels pays less than it would with a leaner launch
  path. This is the first thing to fix to make the GPU numbers meaningful.
* Group size (8 → 32) and placement (earliest/latest) make no measurable difference: the dependences, not
  the cap, bound the groups here.

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

* **Reductions are not compiled.** They run through the stock implementation, which on the GPU means
  copying the reduced dats to the host each step.
* **Data-dependent offsets read every point of the stencil** and select; this is correct for any offset
  inside the declared stencil but a kernel that indexes outside it would be mistranslated rather than
  rejected. (The stock code reads whatever is in memory there, so no valid application does it.)
* **Registered integer constants used as compile-time loop bounds are baked** (they are needed at
  translation time to unroll). A constant whose value changes during the run would go stale; none in
  CloverLeaf do.
* **The GPU launch path is the bottleneck** (see 4.2), so absolute GPU times here say little about the
  fusion gain achievable with a leaner runtime.
* **Sequential regression with fusion** (+18 % in 2D) is unexplained.
* Only the sequential, OpenMP and CUDA backends, in a single process, were exercised; MPI was not tried.
