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
