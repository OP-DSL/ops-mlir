# ops-mlir

An MLIR JIT for the [OPS](https://github.com/OP-DSL/OPS) structured-mesh library. It is an optimising tier that plugs into
OPS's own lazy execution: OPS queues the loops of a program as kernel descriptors and, at a flush, hands them to the JIT, which
fuses them, specialises them and compiles them (CPUs with OpenMP, NVIDIA GPUs with CUDA). Loops the JIT does not take run with
OPS's generated code.

## How it works

1. **Build time.** The OPS translator (`--jit`) runs `ops-kernel-import` over the user kernels (C++ with clang, Fortran with flang) and embeds one
   MLIR artifact per kernel (bytecode and a manifest) next to the ordinary generated code.
2. **Run time.** `ops_par_loop` queues a descriptor instead of running the loop. When the program needs the data, OPS flushes the
   queue through the registered chain executor (`ops_register_chain_executor`).
3. **The executor** (`libopsjit`) claims the loops it supports and lifts them into an `ops.chain` of SSA values. It plans fusion
   groups (`ops-plan`), lowers each group (`ops-jit-cpu`: scf and OpenMP, optionally through the affine dialect;
   `ops-jit-cuda`: NVVM), compiles it with ORC LLJIT and caches the code in memory and on disk, keyed by the bytecode of the group
   and the options.
4. **OPS keeps** the memory, MPI halos, dirty bits, reductions and timing; the JIT only runs loops. In verify mode
   (`OPS_JIT_VERIFY=1`) every group is compared with OPS's own code.

More: [docs/design.md](docs/design.md) (architecture), [docs/user-guide.md](docs/user-guide.md) (building an application, the
environment variables), [docs/passes.md](docs/passes.md) (passes and pipelines, generated),
[docs/performance.md](docs/performance.md) (measurements).

## Layout

| Directory | Contents |
|---|---|
| `core/` | `opdsl`: DSL-independent infrastructure (hashing, artifacts, JIT session, caches, GPU shim). Must not depend on OPS. |
| `ops/` | the `ops` dialect, analyses, transforms, conversions, pipelines, kernel importer, runtime |
| `tools/` | `ops-opt`, `ops-kernel-import`, `ops-jit-replay` |
| `test/` | lit tests (`test/lit`), end-to-end suites |
| `apps/`, `cmake/` | the OPS applications built with the JIT, and the CMake helpers for them |
| `scripts/` | environment (`env.sh`), the build scripts, CI entry point, performance scripts |
| `third_party/OPS/` | git submodule: the OPS sources (branch `feature/jit-executor`), used for the OPS build, the importer tests and the apps |

## Getting the sources

```bash
git clone --recurse-submodules https://github.com/OP-DSL/ops-mlir.git     # or: git submodule update --init third_party/OPS
```

`scripts/env.sh` takes `OPS_SRC` from the submodule when it is checked out; set `OPS_SRC` to use your own OPS clone
(for work on the OPS side). The submodule is pinned to a commit: to move it, `git -C third_party/OPS checkout <commit>`
and commit the new pin. Third-party code is never copied into this repository.

## Build and run

Needs LLVM/MLIR/clang 23.1 with `libomp` (pinned in `cmake/LLVMVersion.cmake`), CMake, Ninja and Python 3. From the
repository root:

```bash
source scripts/env.sh                                          # toolchain and paths; in every new shell
scripts/build_ops.sh                                           # 1. OPS and its translator   (once)
scripts/build_mlir.sh                                          # 2. libopsjit, ops-kernel-import (once)
APPS_CMAKE_ARGS="-DOPSJIT_APPS=laplace2d" scripts/build_apps.sh   # 3. an application (all of them without -DOPSJIT_APPS)

cd $(mktemp -d)
OPS_JIT_STATS=1 $OPSJIT_BUILD_ROOT/OPS-jit-apps/laplace2d_jit  # run with the JIT
OPS_JIT_VERIFY=1 $OPSJIT_BUILD_ROOT/OPS-jit-apps/laplace2d_jit # check every group against OPS's own code
```

The applications are listed in `apps/CMakeLists.txt`. After changing ops-mlir, rerun step 2; after changing an application,
step 3; after changing OPS, all three. For CUDA, add `OPS_CUDA=on` to step 1 and `OPSJIT_APPS_CUDA=ON` to step 3, with
`GPU_ARCH` set to the device's compute capability (e.g. `80`); the executable is `laplace2d_jit_cuda`. Runtime options and how to
build your own application: [docs/user-guide.md](docs/user-guide.md).

The lit tests need `lit` (`pip install -r requirements-test.txt`, or `OPSJIT_LIT` pointing at an `llvm-lit`); without it the
rest still builds and runs.

## Citation

If you use this work, please cite:

```
A. Dascombe, P. Thilakaraj, and G. R. Mudalige, “Code Generation with MLIR for the OPS Structured-Mesh DSL,”
in Proceedings of the 2026 IEEE/ACM 12th Workshop on the LLVM Compiler Infrastructure in HPC (LLVM-HPC 2026),
Chicago, IL, USA, Nov. 16, 2026, to be published.
```

The paper describes the first version of ops-mlir, which is in the history of `main`; this version keeps its idea and replaces
the embedded Python/xDSL lowering with the executor described above.
