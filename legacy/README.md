# ops-mlir
MLIR Compiler Implementation for OPS DSL

## Dependencies

- CMake >= 3.20
- A C++17 compiler
- [LLVM/MLIR](https://github.com/llvm/llvm-project) built from source, with the `mlir` project enabled
- [OPS](https://github.com/OP-DSL/OPS) (the OPS DSL C library)
- OpenMP

## Python Dependencies

- xDSL (for generating OPS DSL code from Python)

## Installation

### 1. Build LLVM/MLIR

Clone and build `llvm-project` with the MLIR project enabled, e.g.:

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON
ninja
```

### 2. Clone OPS

Clone the [OPS](https://github.com/OP-DSL/OPS) repository — `ops-mlir`
builds the OPS sequential backend (`ops_seq`) from its source tree and
links against it.

### 3. Set environment variables

`ops-mlir`'s CMake configuration reads the LLVM/MLIR build/source
locations and the OPS checkout location from environment variables — it
does not hardcode any paths. Set the following before configuring the
project:

```bash
export MLIR_BUILD_DIR=/path/to/llvm-project/build
export LLVM_BUILD_DIR=/path/to/llvm-project/build
export LLVM_SOURCE_DIR=/path/to/llvm-project/llvm
export MLIR_SOURCE_DIR=/path/to/llvm-project/mlir
export OPS_ROOT=/path/to/OPS/ops/c
```

A template is provided in [`env_setup_template`](env_setup_template). Copy it to `env_setup`, edit the
paths in the copy, and `source` it before building:

```bash
cp env_setup_template env_setup
# edit env_setup: set the paths
source env_setup
```

### 4. Build ops-mlir

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
cmake --build build -j$(nproc)
```

The project must be compiled with the clang++ bundled in the LLVM build
(`$MLIR_BUILD_DIR/bin/clang++`, added to `PATH` by `env_setup`). GCC 12 is
incompatible with the generated MLIR headers in LLVM 23. The OpenMP
headers and library are located automatically from `MLIR_BUILD_DIR` — no
extra env vars or `-D` flags are needed.

CMake configuration will fail with a clear error if any of the five
environment variables above are not set.

### 5. Use xDSL Fork
Needed for OPS reductions.

```bash
# Remove standard xDSL (if needed)
pip uninstall xdsl

git clone https://github.com/Archii0/xdsl.git
# you may need to change to the branch "stencil-reduce"
# git checkout stencil-reduce

pip install -e path/to/xdsl-fork

```

TLDR - only partial reduction support is upstreamed into xDSL, waiting on stencil dialect stakeholders to validate my full changes still...

## Usage

Run the 2D Laplace example:

```bash
./build/apps/c/laplace_2d/laplace_2d
```

## Lazy execution and loop fusion

`ops_par_loop` only queues a loop. Queued loops run when something needs their
results, as one compiled module (cached by the shape of the queue):

- `compile_and_execute()` (explicit flush),
- the OPS accessors that expose data to the host (`ops_dat_get_raw_pointer`,
  `ops_dat_fetch_data*`, `ops_dat_set_data*`, `ops_print_dat_to_txtfile`,
  `ops_reduction_result`, `ops_timing_output`, `ops_halo_transfer`,
  `ops_exit`, ...), which flush first and, on the GPU, copy the dat back,
- `sync_all_host_buffers()`, before reading `dat->data` directly,
- the queue reaching `OPS_MLIR_QUEUE_MAX` loops (default 512).

Read-only `ops_arg_gbl` values are captured when the loop is enqueued.

At flush time a planner (`include/runtime/FusionPlanner.h`) splits the queue
into groups that each become one generated kernel. It builds a dependence DAG
over the whole queue (RAW / WAR / WAW between overlapping footprints), so a loop
may be moved ahead of independent loops to join an earlier kernel. Loops fuse
only where no value has to travel between different points: a non-zero-offset
read of something an earlier member wrote (RAW), or a write to something an
earlier member reads at a non-zero offset (WAR, e.g. the Jacobi
`apply_stencil` / `copy` pair), keeps two loops in separate kernels. Loops with
reductions never fuse or move (they are compiled, but each is its own kernel). Loops with different ranges fuse into one launch
over the bounding box; each member then only runs where the point is inside its
own range ("guarded"), provided the box is not larger than the members' summed
volumes and any member smaller than the box reads point-locally.

[`docs/compilation_flow.md`](docs/compilation_flow.md) follows loops of Taylor-Green and CloverLeaf through every
stage (capture, flush, planning, xDSL, kernel translation, the three backends, execution) with the real IR of each.
Design, diagrams and the fusion rules are described in
[`docs/loop_fusion.md`](docs/loop_fusion.md); measurements on the Taylor-Green
vortex are in [`docs/tgv_evaluation.md`](docs/tgv_evaluation.md). CloverLeaf 2D and 3D from the OPS
repository run unmodified and entirely through the JIT (every loop is compiled; a stock-implementation fallback remains
for kernels the translator rejects), with a differential verifier; see [`docs/cloverleaf.md`](docs/cloverleaf.md) and, for the larger decks on the A100 (fusion size,
register pressure, occupancy), [`docs/cloverleaf_large.md`](docs/cloverleaf_large.md).
All of these documents also exist in Simplified Technical English (ASD-STE100) in [`docs/ste/`](docs/ste/README.md).

| Variable | Default | Meaning |
|---|---|---|
| `OPS_MLIR_FUSION` | `1` | `0` disables fusion (one kernel per loop) |
| `OPS_MLIR_FUSION_REORDER` | `1` | `0` only fuses adjacent loops (no dependence DAG) |
| `OPS_MLIR_FUSION_PLACEMENT` | `earliest` | `latest` joins the last legal kernel instead of the first |
| `OPS_MLIR_FUSION_GUARDED` | `1` | `0` only fuses loops with identical ranges |
| `OPS_MLIR_FUSION_MAX` | `8` | maximum loops per fused kernel |
| `OPS_MLIR_FUSION_BOX_RATIO` | `1.0` | bounding box volume / summed member volumes limit |
| `OPS_MLIR_QUEUE_MAX` | `512` | auto-flush at this queue length |
| `OPS_MLIR_PLAN` | unset | print each distinct plan: kernels, traffic estimate, why loops started new kernels |
| `OPS_MLIR_STRICT_FP` | unset | reject single-precision kernels whose body computes in double |
| `OPS_MLIR_DUMP_LOWERED` | unset | print the `ops.par_loop` IR and the per-group lowered IR |
| `OPS_MLIR_STATS` | unset | print loops / launches / flushes / compiles at exit |

## Precision

Kernels and dats may be `float` or `double`. The element type of each dat comes
from the type string given to `ops_decl_dat`, and read-only scalar globals are
`float` or `double` by `sizeof`. A loop's written dats must share a type.
`extern` kernel constants registered with `ops_register_kernel_constant` are
read through the pointer type the kernel header declares. The GPU used in
development is fast in single precision only, so the GPU tests run `float`.

A single-precision Taylor-Green vortex (`opensbli_f32`) is generated from the
double-precision sources at build time by
`apps/c/taylor_green_vortex/to_single_precision.py`: types, type-name strings,
every floating literal, `M_PI` and math calls are converted. Setting
`OPS_MLIR_STRICT_FP=1` makes the kernel translator reject any single-precision
kernel that would silently compute in double.

## Tests

```bash
ctest --test-dir build                      # everything except -L gpu-fp64
ctest --test-dir build -L cpu               # seq / OpenMP
ctest --test-dir build -L gpu               # fp32 on the GPU
ctest --test-dir build -L gpu-fp64          # fp64 on the GPU (correctness only)
```

- `unittests/FusionPlannerTest` and `unittests/KernelIRBuilderTest` need no JIT.
- `tests/e2e/fusion_cases.cpp` is built for `float` and `double`
  (`fusion_cases_f32` / `fusion_cases_f64`). Each case is compared with a host
  reference computed in `double`, fused output is compared with unfused
  output, and the number of launched kernels is asserted. It is also run with
  `OPS_MLIR_FUSION=0` and `OPS_MLIR_FUSION_GUARDED=0`.
- The test kernels live in `tests/kernels/`. `KernelIRBuilder` parses a kernel
  header with fixed compiler flags, so precision is chosen by which thin header
  (`kernels_f32.h` or `kernels_f64.h`) the test registers.

## Citing

```
A. Dascombe, P. Thilakaraj, and G. R. Mudalige, “Code Generation with MLIR for the OPS Structured-Mesh DSL,”
in Proceedings of the 2026 IEEE/ACM 12th Workshop on the LLVM Compiler Infrastructure in HPC (LLVM-HPC 2026),
Chicago, IL, USA, Nov. 16, 2026, to be published.
```
