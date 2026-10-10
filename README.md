# ops-mlir (v2)

An MLIR JIT for the [OPS](https://github.com/OP-DSL/OPS) structured-mesh library, built as an optimising
tier that plugs into OPS's own lazy execution (the kernel-descriptor queue).

This branch (`v2`) is a rewrite in progress. See `docs/PROGRESS.md` for where it stands and
`IMPLEMENTATION_PLAN.md` / `ARCHITECTURE_REVIEW.md` (kept next to the sources in `~/ops-mlir-work`) for the
design. The earlier prototype is in `legacy/` and is not built by default.

## Layout

| Directory | Contents |
|---|---|
| `core/` | `opdsl`: DSL-independent infrastructure (hashing, artifacts, JIT session, caches, GPU shim). Must not depend on OPS. |
| `ops/` | the `ops` dialect, analyses, transforms, conversions, pipelines, kernel importer, runtime |
| `tools/` | `ops-opt`, `ops-kernel-import`, `ops-jit-replay` |
| `test/` | lit tests (`test/lit`), end-to-end suites |
| `cluster/v2/` | SLURM job scripts for the `cl` cluster |
| `scripts/` | environment (`env.sh`), OPS build, CI entry point |
| `third_party/lit/` | vendored `lit` (from llvm-project 23.1; the pinned LLVM install ships FileCheck but no lit) |
| `legacy/` | the previous prototype (xDSL based); removed at the end of the project |

## Build and test

The toolchain is pinned (LLVM/MLIR/clang 23.1.0, `cmake/LLVMVersion.cmake`); `scripts/env.sh` selects it on the
cluster and in WSL.

```bash
source scripts/env.sh
scripts/ci.sh          # configure, build, check-ops-mlir
```

Targets: `check-ops-mlir` (everything), `check-ops-mlir-lit`, `check-ops-mlir-unit`.
