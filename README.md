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
| `third_party/OPS/` | git submodule: the OPS sources (branch `feature/jit-executor`), used for the OPS build, the importer tests and the apps |
| `legacy/` | the previous prototype (xDSL based); removed at the end of the project |

## Getting the sources

```bash
git clone --recurse-submodules https://github.com/OP-DSL/ops-mlir.git     # or: git submodule update --init third_party/OPS
```

`scripts/env.sh` takes `OPS_SRC` from the submodule when it is checked out; set `OPS_SRC` to use your own OPS clone
(for work on the OPS side). The submodule is pinned to a commit: to move it, `git -C third_party/OPS checkout <commit>`
and commit the new pin. Third-party code is never copied into this repository.

## Build and test

The toolchain is pinned (LLVM/MLIR/clang 23.1.0, `cmake/LLVMVersion.cmake`); `scripts/env.sh` selects it on the
cluster and in WSL.

```bash
source scripts/env.sh
scripts/ci.sh          # configure, build, check-ops-mlir
```

Targets: `check-ops-mlir` (everything), `check-ops-mlir-lit`, `check-ops-mlir-unit`.

The lit tests need `lit`, which the pinned LLVM install does not ship: `pip install -r requirements-test.txt`, or
point `OPSJIT_LIT` (environment or `-DOPSJIT_LIT=`) at the `llvm-lit` of an LLVM build. Without it
`check-ops-mlir-lit` fails with a message saying so; the rest builds and runs.
