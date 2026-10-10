//===- Pipelines.h - the named pipelines of the OPS JIT ----------*- C++ -*-===//
//
//   ops-jit-frontend            ops-materialize-kernels, ops-specialize-consts, canonicalize, cse
//   ops-jit-plan                ops-fuse-point-local, ops-forward-stores, canonicalize, cse, ops-outline-groups
//                               options: max-loops reorder placement guarded box-ratio
//   ops-jit-cpu                 ops-specialize, convert-ops-to-scf, OpenMP, down to the LLVM dialect
//                               options: omp specialize soa
//
// They are registered with the process-wide opdsl::PipelineRegistry (the runtime builds them by name and option
// string) and, for ops-opt, as named pipelines on the command line (-ops-jit-cpu="omp=false").
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_PIPELINES_PIPELINES_H
#define OPS_MLIR_PIPELINES_PIPELINES_H

#include "opdsl/Pipeline/PipelineRegistry.h"

namespace ops_mlir::ops {

/// Adds the pipelines of this library to `registry`.
void registerOPSPipelines(opdsl::PipelineRegistry &registry = opdsl::PipelineRegistry::global());

/// Makes the pipelines of the global registry available as named pipelines of mlir-opt style tools. Call after
/// registerOPSPipelines().
void registerOPSPipelinesWithMLIR();

} // namespace ops_mlir::ops

#endif // OPS_MLIR_PIPELINES_PIPELINES_H
