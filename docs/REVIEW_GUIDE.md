# Review guide

What is on the two branches, what changed in OPS and why, what is not done, and how to check the numbers. Nothing here is merged and nothing has
been pushed to GitHub: the branches live in the local clones and in staging repositories on the `cl` cluster (`~/ops-jit-cl/{OPS,ops-mlir}.git`).

| Repository | Branch | Base | Commits |
|---|---|---|---|
| `OP-DSL/OPS` | `feature/jit-executor` | `develop` @ `c0af0f124` (0 behind at the time of writing) | 16, all additive |
| `OP-DSL/ops-mlir` | `v2` | `main`, after the earlier `feature/fusion` squash (`9628feb`) | ~100 |

Documents: `docs/user-guide.md` (use), `docs/design.md` (architecture), `docs/PROGRESS.md` (the log of what was found, in order),
`docs/performance.md`, `docs/baseline.md` (OPS before and after), `docs/coverage.md` (importer), `docs/cir-spike.md` (frontend B), and the plan and
review in `IMPLEMENTATION_PLAN.md` / `ARCHITECTURE_REVIEW.md` next to the repositories.

## 1. OPS side (`feature/jit-executor`)

All changes are inactive unless an executor is registered or `--jit` is used; `OPS_LAZY_EXECUTOR_TEST=split|manual` runs any application through
the executor interface with no JIT. The existing OPS tests give the same results as before (`docs/baseline.md`: 73/73 CPU, the GPU failures are the
ones that exist without the branch).

| Commit | Change | Why |
|---|---|---|
| `a0f390f9f` P0.3-P0.5 | `ops_lazy_executor.h`: `ops_register_chain_executor` (a function that receives the queued loops of a run), queue-only mode, `ops_jit_artifact` pointer in the kernel descriptor | the hook the JIT plugs into; the executor decides per run which loops it takes (`ops_lazy_execute_default` runs the rest) |
| `697580e04` P0.6 | every call that observes data on the host (`ops_reduction_result`, `ops_dat_fetch_*`, halo/partition calls, ...) flushes the queue; `apps/c/lazy_triggers` tests each trigger | a chain executor must not leave work queued across an observation point |
| `fc613577f` P0.7/P0.8 | registry of the constants (`ops_decl_const`, `ops_update_const`) with a snapshot of the value; built-in test executors | the JIT binds constants as arguments, or folds them into the code (`bound`); `ops_update_const` invalidates what was folded |
| `23eedac9f`, `a0017fc02` | `lazy_triggers`: one output file per process; no loop that is empty on some MPI ranks under tiling | test hygiene (ctest runs the variants concurrently; an OPS tiling limitation) |
| `f89e004d3`, `4537a57d2`, `2ee69f1e8` | translator `--jit`: enqueue with the artifact, write `jit_kernels.json`, `ops_jit_init` with C linkage, import in the translation unit that sees the kernel | build-time half of frontend A |
| `f2f688c4f` | `ops_lazy_dat_shift`, `ops_lazy_reduction_slot` | the base offset of a dat and the slot of a reduction handle, for executors that index dats themselves |
| `1bf66d9f3` | `ops_register_chain_executor` accepts the default instance (NULL) | `libopsjit` must not need OPS objects that a static link leaves out |
| `9a59591e6`, `a4808de00`, `6e480d3a5` | translator: token-less declaration references (OpenSBLI), `ops_init(` found at any indentation | TGV (OpenSBLI) and CloverLeaf layouts |
| `b44737a17` | `ops_enqueue_kernel`: the edge-dat rule (flush and run eagerly) applies only with more than one process | its dirty bit is cleared by the MPI exchange only: on one process every loop that read an edge dat after a write flushed the queue (CloverLeaf lost 2.5x, `performance.md`); also benefits OPS's own tiling |
| `b923f05c5` | `half` is `_Float16` under clang | the host side of the half precision applications did not compile |
| `c5ef88f9c` | CMake requires a Fortran compiler only when `BUILD_OPS_FORTRAN` | build hygiene on machines without one |

## 2. ops-mlir `v2`: the commits by phase

| Phase | Commits (first .. last) | Content |
|---|---|---|
| P0 groundwork | `29d4ba3` .. `a8cd88e` | pinned LLVM 23.1, environment scripts, building OPS and its apps on the cluster |
| P1 core, dialect, CPU lowering | `5e0979c` .. `0097681` | the `ops` dialect, footprints, dependence analysis, the fusion planner, materialise/specialise/fuse/forward/outline passes, `convert-ops-to-scf`, `opdsl::JITSession`, `PersistentCache`, `PipelineRegistry` |
| P2 frontend A | `61548d1` .. `9bfcc43` | `ops-kernel-import` (clang AST walker), per-kernel artifacts, CMake wiring, coverage over all applications |
| P3 runtime | `6bd3b23` .. `f1a8515` | `libopsjit`: claim, lifter, plans, group compiler and caches, verify mode, replay, per-run tests of every application |
| P4 GPU | `1c0a844` .. `3b1198b`, `ad5b713` | CUDA shim, `ops-jit-cuda`, device data and reductions, per-group fallback |
| P7 items done | `b799722` (printf), `1a40a2f` (half), `1e41754` .. | printf as `ops.print`, half precision, persistent plan cache, `ops-fp-contract`, kernel `noalias` |
| P5 ClangIR spike | `766430c`, `scripts/cir/` | the CIR build job, the corpus script, `docs/cir-spike.md` |
| Performance work | `f965f2e` .. | the measurements and fixes of `docs/performance.md` |

## 3. Things that go beyond the plan's normative interfaces (section 4)

`ops.call` and `ops.print`; `FusionPlanner.h` (the DAG planner as a library); the loop-unrolling policy of the importer; the relaxed number of
components of a kernel accessor; `--source` translation-unit mode of the importer; `OPS_ACC_IGNORE`; `f16`; the GPU reduction buffer ABI
(one partial value per warp in a device buffer after the scalars); `ops_lazy_dat_shift`, `ops_lazy_reduction_slot`; a NULL instance in
`ops_register_chain_executor`; `PackedArgs` (the argument block of a group call), `GroupCompiler` and `OPSMLIRRuntimeCore` (the runtime without OPS,
for replay and tests); `OPS_JIT_FMA`, `OPS_JIT_VERIFY_{FLOAT,HALF}_TOL`; the persistent plan cache; `ops-fp-contract`, `ops-gpu-kernel-attrs`.

## 4. State of the acceptance criteria (plan section 9)

* **Apps pass.** laplace2d, poisson, lazy_triggers, CloverLeaf 2D/3D, TeaLeaf, lowdim and TGV DP/SP/SPDP/HP/HPSP pass in verify mode (every group compared
  with OPS's stock code) on: CPU (one process and `mpirun -np 4`), A100 (CUDA), CUDA with MPI (np=1, and np=2 on the two A100s for CloverLeaf 2D/3D and
  TeaLeaf). The suite is `ctest -L jit-app` in the application build trees (`scripts/build_apps.sh`); `docs/PROGRESS.md` has the history. HPSP verify only catches
  gross errors (the host code rounds a half precision expression once, the JIT every operation). **Not run:** the V100 node, CUDA-aware MPI.
* **Last full run** (cluster, the final commit of `v2`; `ctest -L jit-app`): CPU nodes 58 of 58 application tests (one process and `mpirun -np 4`; default, no fusion, verify, cache)
  and 10 of 10 TGV tests; A100 38 of 38 and 10 of 10 (CUDA, MPI+CUDA np=1); `OPS_JIT_SPECIALIZE=shapes` and `=ranges` pass the same suites (one parallel-ctest flake of
  `lowdim_jit_verify`, concurrent HDF5 files, passes alone). `ninja check-ops-mlir-lit` 47, `check-ops-mlir-unit` 12.
* **Claim rate.** CloverLeaf 2D 13623 of 13625 loop executions, 3D 52568 of 52577, TGV all; with MPI the loops that use 1-D coordinate dats stay with OPS
  (94% in 2D, 97% in 3D at np=2) because OPS exchanges those dats itself.
* **Performance** (`docs/performance.md`): CPU: CloverLeaf at parity (2D 1.01x, 3D 0.96x of AOT time), TGV 4-10x faster; A100: within 6-10% of AOT
  CUDA wall time (host start-up), per kernel within 4%. The plan's GPU target "fused at least as fast as AOT" is **not met** for CloverLeaf.
* **Compile time.** Cold CloverLeaf 2D 15-17 s on the 16-thread development machine (target 10 s, **not met**); warm runs read the disk cache (3 ms per
  group) and the persistent plan cache.
* **Tests.** `ninja check-ops-mlir-lit` 45 tests, `ninja check-ops-mlir-unit` 12 tests (including real-GPU tests on a machine with a GPU).
  `scripts/ci.sh` runs the local checks (build, lit, unit tests, the core independence check); the cluster scripts are `cluster/v2/{build,ctest,perf}.sbatch`.

## 5. Known limitations and open declines

* Importer (frontend A): `docs/coverage.md`: 588 of 626 kernels of all OPS applications import; the rest: initialiser lists (11), accessor `combine_*` reductions,
  `unsigned short`/`char`/complex dats, a computed accessor component (lbm), writes to globals, `bfloat16`. They run with OPS.
* Loops with multigrid stencils, `isdevice` loops without a usable CUDA device, edge dats under MPI, stride-0 writes in loops of more than one point,
  reductions written as `*r = v` in loops of more than one point: run with OPS (reason: `OPS_JIT_EXPLAIN=1`).
* Not done from the plan: P6 (LLVM IR import as a fallback), P7.1 reduction fusion, P7.2 dead temporaries, P7.4-P7.11 (register-aware fusion,
  guarded fusion, tiling interface, OPS tiler bridge, asynchronous compilation, autotuning, direct OpenSBLI emission, more targets), P8 (op2-mlir on
  `opdsl`), P9 (HIP, cannot be tested here). `core/` has no OPS dependency (checked by a CMake test), which is the part P8 needs.
* Frontend B (ClangIR): corpus study and design only, see `docs/cir-spike.md`.

## 6. Decisions that are yours

1. **A or B.** Recommendation in `docs/cir-spike.md`.
2. **Retiring `legacy/`, the xDSL files and the old docs** (P4.8): not touched. About 1100 `*:Zone.Identifier` files (Windows download markers) sit untracked in the
   ops-mlir tree; they are not committed (one commit that contained 101 of them was undone before anything left the machine). They need your OK to be deleted.
3. **Pushing** the two branches to GitHub (never merged, never to `develop`/`main`).
4. The interface additions of section 3.

## 7. Reproducing the numbers

Environment: `source scripts/env.sh` (local WSL or the cluster). Build: `scripts/build_ops.sh`, `scripts/build_mlir.sh`, `scripts/build_apps.sh`; cluster:
`cluster/v2/sync.sh` then `sbatch cluster/v2/build.sbatch all`. Tests: `ctest --test-dir <apps build> -L jit-app` (CPU nodes: `-E cuda`; GPU nodes: `-R cuda`).
Measurements: `scripts/perf/*.sh` through `cluster/v2/perf.sbatch` (the list is at the top of `docs/performance.md`); raw results were in
`~/ops-jit-cl/results/`.
