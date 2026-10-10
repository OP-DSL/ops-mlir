# Implementation progress log

Running log of `IMPLEMENTATION_PLAN.md` execution, kept for the final review. One entry per task:
what was done, deviations from the plan, and facts discovered that the plan did not know.
Newest entries at the bottom. Cluster = `cl` (SLURM); local = WSL Ubuntu 24.04.

## Environment facts (2026-10-10)

- Local `llvm-project/build` (23.1.2, main snapshot) is **unusable**: it was copied through Windows, so all
  executables lost their exec bit and the 32 symlinks in `bin/` and `lib/` (clang, clang++, ...) became
  0-byte files. It is not used by v2. (`chmod +x` fixed `mlir-opt`, but not the symlinks.)
- The `ops-mlir-work` source trees (OPS, ops-mlir, xdsl, llvm-project) carry `*:Zone.Identifier` files and
  broken refs `refs/heads/*:Zone.Identifier` from the same Windows copy. They are ignored through
  `.git/info/exclude`; deleting them is pending Istvan's OK (plan P0.1).
- The local OPS clone was shallow (1 commit); it was un-shallowed from `origin` so the final review has a history.
- The Bash tool of the harness is Git Bash on Windows; WSL commands go through `wsl.exe` (PowerShell).
  Scripts piped to `bash -s` get a trailing `\r` on the last line from PowerShell: end them with a sentinel.
- Local machine: 16 cores, 15 GB RAM, RTX 4050 (sm_89, driver visible in WSL), `nvcc` 12.0, no MPI, no
  `gfortran`, no `sudo`. MPI/HDF5 tests therefore run on the cluster.

## P0.0 Toolchain pin and cluster bootstrap

- **Pin:** LLVM/MLIR/clang **23.1.0** = the existing cluster install
  `~/MLIR/lib_install/llvm-project-23.1.0-CUDA-Release` (assertions on, `X86;NVPTX`, includes lld, the LLVM
  OpenMP runtime `lib/x86_64-unknown-linux-gnu/libomp.so`, `libmlir_cuda_runtime.so`, FileCheck/not/count,
  mlir-runner; no `llvm-lit`). It was rsync-copied (5.4 GB) to `~/ops-mlir-work/toolchain/llvm-23.1.0`, so
  local and cluster use byte-identical toolchains. `cmake/LLVMVersion.cmake` fails the configure on a
  different release series.
- `scripts/env.sh` detects local vs cluster and exports the same variables on both (`LLVM_DIR`, `MLIR_DIR`,
  `CC/CXX`, `OPS_SRC`, `OPS_INSTALL`, CUDA/MPI/HDF5 paths). On the cluster it enables gcc-toolset-13 (libstdc++ the
  clang binaries need) and loads `openmpi/4.1.6` and `OP2-env` (CUDA 12.3, HDF5, MPI paths).
- Cluster: `~/ops-jit-cl/{OPS.git,ops-mlir.git,work,logs,results}`; local remotes `cl-ops`, `cl-mlir`;
  `cluster/v2/sync.sh` pushes the feature branches (never forced) and fast-forwards the work trees.
  SSH key copied into WSL (`~/.ssh/id_rsa_cl` + `Host cl` entry, with Windows CRLF line endings converted).
- **Acceptance:** `cluster/v2/hello.sbatch` (pinned clang + OpenMP/libomp + MPI + CUDA-driver check)
  passes on a CPU node (job 223629: `omp threads=4`, `mpi ranks=2 allreduce=2`, `HELLO OK`).
  The A100 (223630) and V100 (223631) jobs waited behind other users; see below.
- **Deviation from the plan:** the plan's acceptance said "builds the OPS libraries with the pinned clang"; the
  hello job checks the toolchain only, and the OPS build is P0.2 (`scripts/build_ops.sh`).
- Notes: `.gitignore` ignored `*.cmake`; `!cmake/*.cmake` added. `core.fileMode=false` in OPS and ops-mlir
  (Windows-copy mode noise), so exec bits are recorded with `git update-index --chmod=+x`.
  A100 (223630) and V100 (223631) hello jobs also pass (`Tesla V100-PCIE-16GB sm_70`, `NVIDIA A100-PCIE-40GB sm_80`):
  **P0.0 accepted** on CPU, V100 and A100 nodes.

## P0.1 Hygiene

- `core.fileMode=false` in OPS and ops-mlir; `*:Zone.Identifier` ignored through `.git/info/exclude`;
  `apps/c/TeaLeaf/tea.in` restored to its symlink; OPS un-shallowed.
- **Pending Istvan's OK:** deleting the `*:Zone.Identifier` files and the broken `refs/heads/*:Zone.Identifier`
  refs (they only make git print warnings in ops-mlir).

## P0.2 OPS build and baseline

- `scripts/build_ops.sh` builds and installs OPS with the pinned clang (single source of truth, local and cluster);
  `OPS_BUILD_APPS=ON` builds the sample apps and their ctest tests in a separate tree.
  First library build on the cluster (CUDA sm_70+sm_80, MPI, HDF5): clean on the first attempt.
- OPS `CMakeLists.txt`: `project(OPS C CXX)`; Fortran only with `BUILD_OPS_FORTRAN=ON` (it made every
  configure fail on machines without a Fortran compiler; local WSL has none).
- **Local limitation:** no MPI, HDF5 or gfortran in WSL and no sudo, and the system Python has no `venv`/`pip`.
  `scripts/setup_translator_local.sh` builds the translator venv without pip and copies the cluster venv's
  site-packages (same Python 3.12; pure Python + libclang 16.0.6 wheel). Local builds are libs without MPI/HDF5/CUDA;
  `scripts/ops_app_manual.sh` translates and builds one app by hand (OPS's CMake app machinery runs the translator
  over every sample app at configure time, 10+ minutes). MPI, HDF5 and GPU tests run on the cluster.
- **Cluster HDF5 trap:** the shared `op-dsl/hdf5` was relocated; its `h5pcc` wrapper still says
  `prefix=/home/reguly/projects/tests/hdf5`, where an Intel-built HDF5 lives that cannot be linked (needs `libimf`).
  CMake's FindHDF5 trusts the wrapper, so `build_ops.sh` uses a copy of the wrapper with the prefix corrected
  (build-tree only; the shared files are untouched). Also: CMake package registries are disabled for reproducibility.

## P0.3-P0.5 executor hook, queue-only mode, artifact pointer (OPS `a0f390f9f`)

- New headers `ops_lazy_executor.h`, `ops_jit_artifact.h`; descriptor field `jit_artifact`,
  `create_kerneldesc_and_enque_jit()`; `OPS_instance` fields `lazy_executor`, `lazy_executor_ctx`, `lazy_queue`.
- **Deviations from the plan's §4.1:** the API is C++ (not `extern "C"`: `OPS_instance` is a C++ class);
  `ops_lazy_execute_default()` and the per-loop steps are as specified. Additions: `ops_lazy_queue_enabled()`,
  `ops_lazy_transfer_bytes()`; `ops_lazy_account()` also increments the call count (the generated stub does that
  for stock execution). `ops_lazy_prepare()` books its own MPI time. `ops_register_chain_executor()` flushes the
  queue first. `ops_execute()` takes the queue before executing it, so loops enqueued by an executor go to the next one.
- The stock eager path now uses the same helpers (behaviour unchanged: laplace2d eager / OPS_LAZY_QUEUE / OPS_TILING
  give the same results). Fixes on the way: one descriptor-free function instead of three copies; the eager path no
  longer leaks the descriptor of a loop with no points on this rank.

## P0.6 flush triggers (OPS `697580e04`)

- Confirmed by the new `apps/c/lazy_triggers` test against the unfixed library: `ops_print_dat_to_txtfile` and
  queued loops at `ops_exit` are wrong with OPS_TILING/OPS_LAZY_QUEUE. Also fixed from reading: `ops_timing_output`
  (queued loops were missing from the statistics: laplace2d showed `copy 99` instead of 100), MPI
  `ops_dat_get_raw_pointer`, device single-node `ops_halo_transfer` (these two need MPI/GPU: exercised on the cluster).
- `ops_get_data()` flushes: this covers print-to-file, NaN check and the HDF5 writers on both backends.
- `ops_execute()` ignores nested requests while a flush is running.

## P0.7 constants, P0.8 test executors (OPS `fc613577f`)

- Constant registry: every `ops_decl_const`/`ops_update_const` records a snapshot + version (`ops_find_const`);
  a changed value flushes the queue first. **Limitation (documented in the header):** stock generated CPU kernels read
  the *live* variable when they finally run, so only snapshot users (the JIT) get enqueue-time values; the host
  assignment precedes the `ops_update_const` call, so this cannot be fixed on the OPS side for CPU AOT code.
- `OPS_LAZY_EXECUTOR_TEST=split|manual[,seed=N]` (argv or environment): built-in executors that must not change any
  results; `split` hands the queue to the default path in random pieces of 1-5 loops. laplace2d and lazy_triggers pass in
  every combination with/without OPS_TILING locally.

## P1.1-P1.2 skeleton and the `ops` dialect (ops-mlir `36a6c2d`)

- `v2` skeleton with `core/` (opdsl: hashing, test harness, independence check), vendored lit, `check-ops-mlir[-lit|-unit]`,
  `scripts/ci.sh`. The dialect: types, access modes, stencil attr, kernel/chain/par_loop and the body ops, verifiers
  (8 lit tests, ~48 negative cases). **Deviation:** no `StencilAccessOpInterface` yet; passes use `ParLoopOp` accessors
  (`getAccess`, `getStencil`) directly. Add the interface if a second op ever needs it.

## P1.3 footprints and chain dependence

- `Analysis/Footprint.h`: boxes, `Reach` (min/max offset per dimension, strides other than 1 make a dimension unbounded,
  point-locality) and `footprint(range, reach)`. Exact reach: a stencil with only `+1` offsets does not imply the zero offset.
- `Analysis/ChainDependence.h`: `extractLoops(ChainOp)` reads concrete ranges from `ops.range_values` (one dense i64
  array per `par_loop`); `ChainDependence` holds the loops and the RAW/WAR/WAW edges with `fusable` flags and prints them.
  **Write footprints:** the offsets of the `ops.write`s of a materialised body; the whole stencil of the operand when the
  body is empty or never writes it. A non-zero write offset (or stride other than 1) makes every edge on that dat blocking.
- `Analysis/FusionPlanner.h` (**addition to the layout of plan §3**, the planner is pure analysis and `ops-fuse-point-local`
  will apply it): port of the legacy consecutive and DAG planners, `blockedBy`, `numReordered`, `digest`, traffic estimate.
  Guarded fusion treats a member with non-local writes like one with non-local reads.
- Passes `-ops-print-chain-dependence` and `-ops-print-fusion-plan{max-loops reorder placement guarded box-ratio}` (stderr).
- `KernelOp`/`ChainOp` got a `(name, FunctionType)` builder and `addEntryBlock()`.
- Tests (`ops/unittests`): every case of the legacy suite on `ops.chain` (`ChainBuilder.h` builds a chain from compact
  loop specs), new tests for writes at non-zero offsets (TGV boundary pattern, both orders, disjoint footprints, narrowing by
  the body), randomised oracle with and without non-zero write offsets over materialised and unmaterialised bodies. The
  oracle was checked by mutation: dropping the non-zero-write rule or the WAR rule fails it (280 mismatches).
  lit: `test/lit/Analysis/chain-dependence.mlir` (Jacobi, point-wise, reordering, boundary kernel).

## P1.4 `ops-materialize-kernels`, `ops-specialize-consts`

- `Transforms/Passes.td` (tablegen pass declarations, `registerOPSTransformsPasses()`), library `OPSMLIRTransforms`.
- **Materialise:** accessor and reduction parameters map to the loop's dat and reduction block arguments by kind and order;
  a `!ops.gbl<T, n>` parameter takes the next n scalar operands (`gbl_read %g[c]` -> scalar argument); `const_ref` without
  index -> one chain argument per (name, offset, type), appended to the chain signature and recorded in the chain attribute
  `ops.binding_consts` (`[{name, offset, arg}]`), passed to every loop that uses it. The loop is verified afterwards (offsets
  against stencils). Constants with a dynamic index stay as `ops.const_ref` for specialisation.
- **Specialise:** inputs are chain attributes `ops.const_values` (name -> `array<i8: ...>`, the bytes `ops_decl_const`
  registered, little endian) and `ops.const_roles` (name -> "bound"); `names=` overrides the roles. Bound `binding_consts`
  arguments are replaced by constants and removed from the chain signature (other entries renumbered); `scf.for` loops with a
  static trip count <= `max-trip-count` that index a bound constant dynamically are fully unrolled; `const_ref` with a constant
  index folds to the value at `offset + index * array_stride`; arith/scf canonicalisation follows. A remaining dynamic
  `const_ref` is an error. **Deviation:** only loops containing such a reference are unrolled (the plan says all loops with
  constant trip count <= 256, which would also unroll unrelated loops).
- The greedy driver hoists the folded constants out of loop bodies to the chain body (par_loop regions are not isolated).
- lit: `Transforms/{materialize-kernels,invalid-materialize,specialize-consts,invalid-specialize}.mlir`.

## P1.5 `ops-fuse-point-local`, `ops-forward-stores`, `ops-outline-groups`

- **Fuse:** plans with `planFusion` (unguarded) and rewrites: one `par_loop` per group (dat operands united by SSA value,
  access modes joined: read+write -> rw, all inc -> inc; stencil points united; scalar operands united; range operands of the
  first member, which is right because the plan is keyed by the concrete ranges, equal inside a group), bodies cloned in
  member order; loops move before the operation that follows their *segment* (the loops between two reductions), so
  reductions and the operations that use their results never move; `ops.range_values` is rewritten; fused loops carry
  `ops.fused_loops` (member loop ids). Unmaterialised loops are an error. **Not implemented: `guarded=true`** (the pass
  reports an error): it needs a loop-coordinate op for the range guards (`ops.index` is the global index, which differs
  from the iteration coordinate under MPI). The planner supports guarded groups already. Decide in P7 whether TGV benefits.
- **Forward stores:** write -> read of the same (accessor, offsets, component) inside one block, invalidated by nested regions
  that write the accessor; skipped when the dat appears through two operands of the loop. Then narrows modes: rw never read
  -> write, rw never written -> read (so fused intermediates are stored, not loaded).
- **Outline:** `ops.chain @group_<n>` per loop (arguments = distinct operand values in operand order: ranges, dats, scalars;
  captured constants copied in; `ops.range_values` entry carried over); the original chain keeps `ops.call`s. **Addition to
  the dialect:** `ops.call` (symbol user, verifies the callee is an `ops.chain` of matching signature), because the plan
  says "list of calls" and `func.call` does not accept an `ops.chain`.
- lit: `Transforms/{fuse-point-local,forward-stores,outline-groups}.mlir`.
