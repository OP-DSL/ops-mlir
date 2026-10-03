# ops-mlir

> This is the Simplified Technical English (ASD-STE100) version of the project [README.md](../../README.md). Code, listings and numbers are the same as in the original.

An MLIR compiler for the OPS DSL.

## Dependencies

- CMake >= 3.20
- A C++17 compiler
- [LLVM/MLIR](https://github.com/llvm/llvm-project). Build it from source with the `mlir` project enabled.
- [OPS](https://github.com/OP-DSL/OPS) (the OPS DSL C library)
- OpenMP

## Python Dependencies

- xDSL (to generate OPS DSL code from Python)

## Installation

### 1. Build LLVM/MLIR

Clone `llvm-project`. Build it with the MLIR project enabled. For example:

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
ninja
```

### 2. Clone OPS

Clone the [OPS](https://github.com/OP-DSL/OPS) repository. `ops-mlir` builds the OPS sequential backend (`ops_seq`) from the source tree of OPS. It links with this backend.

### 3. Set environment variables

The CMake configuration of `ops-mlir` reads the build and source locations of LLVM/MLIR from environment variables. It also reads the location of the OPS repository from an environment variable. It does not hardcode paths. Before you configure the project, set these variables:

```bash
export MLIR_BUILD_DIR=/path/to/llvm-project/build
export LLVM_BUILD_DIR=/path/to/llvm-project/build
export LLVM_SOURCE_DIR=/path/to/llvm-project/llvm
export MLIR_SOURCE_DIR=/path/to/llvm-project/mlir
export OPS_ROOT=/path/to/OPS/ops/c
```

The file [`env_setup_template`](../../env_setup_template) is a template. Copy it and edit it for your own paths. Then `source` it before you build:

```bash
source env_setup
```

### 4. Build ops-mlir

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
cmake --build build -j$(nproc)
```

You must compile the project with the clang++ that is in the LLVM build (`$MLIR_BUILD_DIR/bin/clang++`). `env_setup` adds it to `PATH`. GCC 12 is not compatible with the generated MLIR headers in LLVM 23. CMake finds the OpenMP headers and library automatically from `MLIR_BUILD_DIR`. You do not need more environment variables or `-D` flags.

If you do not set one of the five environment variables above, the CMake configuration fails with a clear error.

## 5. Use the xDSL Fork

You need the fork for OPS reductions.

```bash
# Remove standard xDSL (if needed)
pip uninstall xdsl

git clone https://github.com/Archii0/xdsl.git
# you may need to change to the branch "stencil-reduce"
# git checkout stencil-reduce

pip install -e path/to/xdsl-fork

```

Short version: upstream xDSL has only a part of the reduction support. The author waits for the stencil dialect stakeholders to validate the full changes.

## Usage

Run the 2D Laplace example:

```bash
./build/apps/c/laplace_2d/laplace_2d
```

## Lazy Run of Loops and Loop Fusion

`ops_par_loop` only puts a loop in the queue. The queued loops run when something needs their results. They run as one compiled module. The cache key of the module is the shape of the queue.

The queue runs in these cases:

- You call `compile_and_execute()` (an explicit flush).
- You call an OPS accessor that gives data to the host. The accessors are `ops_dat_get_raw_pointer`, `ops_dat_fetch_data*`, `ops_dat_set_data*`, `ops_print_dat_to_txtfile`, `ops_reduction_result`, `ops_timing_output`, `ops_halo_transfer`, `ops_exit` and other such accessors. These accessors flush first. On the GPU, they also copy the dat back.
- You call `sync_all_host_buffers()`. Do this before you read `dat->data` directly.
- The queue reaches `OPS_MLIR_QUEUE_MAX` loops. The default is 512.

The runtime captures the read-only `ops_arg_gbl` values when it puts the loop in the queue.

At flush, the planner (`include/runtime/FusionPlanner.h`) splits the queue into groups. Each group becomes one generated kernel. The planner builds a dependence DAG over the whole queue. The DAG has RAW, WAR and WAW edges between overlapping footprints. Because of this, the planner can move a loop ahead of independent loops. The loop then joins an earlier generated kernel.

Loops fuse only where no value has to travel between different points. Two cases keep two loops in separate generated kernels:

- A loop reads a dat at a non-zero offset, and an earlier member wrote the dat (RAW).
- A loop writes a dat, and an earlier member reads the dat at a non-zero offset (WAR). The Jacobi `apply_stencil` / `copy` pair is an example.

Loops with reductions never fuse, and the planner never moves them. The runtime compiles them, but each loop has its own generated kernel.

Loops with different ranges can fuse into one launch over the box. Then each member runs only where the point is in its own range. The member is "guarded". This is true only if two conditions are true:

1. The box is not larger than the sum of the volumes of the members.
2. Each member that is smaller than the box reads only at its own point.

The document [`docs/compilation_flow.md`](compilation_flow.md) follows loops of Taylor-Green and CloverLeaf through every stage. The stages are capture, flush, plan, xDSL, translation of the user kernel, the three backends and run. The document shows the real IR of each stage.

The document [`docs/loop_fusion.md`](loop_fusion.md) describes the design, the diagrams and the fusion rules. The document [`docs/tgv_evaluation.md`](tgv_evaluation.md) has measurements on the Taylor-Green vortex.

CloverLeaf 2D and 3D from the OPS repository run without changes. They run fully through the JIT: the runtime compiles every loop. A fallback to the stock implementation remains for the user kernels that the translator rejects. A differential verifier checks the result. See [`docs/cloverleaf.md`](cloverleaf.md). For the larger decks on the A100 (fusion size, register pressure, occupancy), see [`docs/cloverleaf_large.md`](cloverleaf_large.md).
These documents also exist in the original English in [`docs/`](..).

| Variable | Default | Meaning |
|---|---|---|
| `OPS_MLIR_FUSION` | `1` | `0` turns fusion off (one generated kernel for each loop) |
| `OPS_MLIR_FUSION_REORDER` | `1` | `0` fuses only adjacent loops (no dependence DAG) |
| `OPS_MLIR_FUSION_PLACEMENT` | `earliest` | `latest` joins the last legal generated kernel and not the first |
| `OPS_MLIR_FUSION_GUARDED` | `1` | `0` fuses only loops with identical ranges |
| `OPS_MLIR_FUSION_MAX` | `8` | maximum number of loops in one generated kernel |
| `OPS_MLIR_FUSION_BOX_RATIO` | `1.0` | limit for the volume of the box divided by the sum of the member volumes |
| `OPS_MLIR_QUEUE_MAX` | `512` | the runtime flushes automatically at this queue length |
| `OPS_MLIR_PLAN` | unset | print each different plan: the generated kernels, the traffic estimate, and why loops started new generated kernels |
| `OPS_MLIR_STRICT_FP` | unset | reject single-precision user kernels that compute in double |
| `OPS_MLIR_DUMP_LOWERED` | unset | print the `ops.par_loop` IR and the lowered IR of each group |
| `OPS_MLIR_STATS` | unset | print the number of loops, launches, flushes and compiles at exit |

## Precision

Kernels and dats can be `float` or `double`. The type string that you give to `ops_decl_dat` sets the element type of each dat. Read-only scalar globals are `float` or `double` by `sizeof`. The written dats of a loop must have the same type.

The runtime reads `extern` kernel constants that you register with `ops_register_kernel_constant` through the pointer type that the kernel header declares. The GPU in the development system is fast only in single precision, so the GPU tests run `float`.

The build makes a single-precision Taylor-Green vortex (`opensbli_f32`) from the double-precision sources. The script `apps/c/taylor_green_vortex/to_single_precision.py` does this. It changes the types, the type-name strings, every floating literal, `M_PI` and the math calls. When you set `OPS_MLIR_STRICT_FP=1`, the kernel translator rejects each single-precision user kernel that silently computes in double.

## Tests

```bash
ctest --test-dir build                      # everything except -L gpu-fp64
ctest --test-dir build -L cpu               # seq / OpenMP
ctest --test-dir build -L gpu               # fp32 on the GPU
ctest --test-dir build -L gpu-fp64          # fp64 on the GPU (correctness only)
```

- `unittests/FusionPlannerTest` and `unittests/KernelIRBuilderTest` do not need the JIT.
- The build makes `tests/e2e/fusion_cases.cpp` for `float` and `double` (`fusion_cases_f32` / `fusion_cases_f64`). The test compares each case with a host reference that it computes in `double`. It compares the fused output with the unfused output. It asserts the number of launched generated kernels. The test also runs with `OPS_MLIR_FUSION=0` and `OPS_MLIR_FUSION_GUARDED=0`.
- The test kernels are in `tests/kernels/`. `KernelIRBuilder` parses a kernel header with fixed compiler flags. Because of this, the test chooses the precision with the thin header that it registers (`kernels_f32.h` or `kernels_f64.h`).

## Citing

```
A. Dascombe, P. Thilakaraj, and G. R. Mudalige, “Code Generation with MLIR for the OPS Structured-Mesh DSL,”
in Proceedings of the 2026 IEEE/ACM 12th Workshop on the LLVM Compiler Infrastructure in HPC (LLVM-HPC 2026),
Chicago, IL, USA, Nov. 16, 2026, to be published.
```
