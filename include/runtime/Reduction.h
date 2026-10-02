//===- Reduction.h - Folding per-point contributions to one value --------===//
//
// A loop with a reduction argument does not accumulate in place. Its kernel returns this
// point's contribution (starting from the operation's identity), the generated function stores
// it in a scratch buffer over the iteration box, and one of the functions below folds that
// buffer to a single value. The caller combines the result with the application's reduction
// handle.
//
// The fold is deterministic: the buffer is cut into a fixed number of chunks (on the host) or
// strided over a fixed number of threads and blocks (on the GPU), each accumulated in order, and
// the partial results are combined in order -- the same value for any number of threads.
//
// These are plain C++ and PTX rather than generated MLIR: they are tiny, fixed, and needed on
// every backend, and a JIT-compiled helper engine per element type and operation was not worth
// what it cost.
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_RUNTIME_REDUCTION_H
#define OPS_MLIR_RUNTIME_REDUCTION_H

#include "runtime/Core.h"

#include <cstddef>
#include <cstdint>

#ifdef OPS_ENABLE_CUDA
#include <cuda.h>
#endif

namespace ops_mlir {

/// Identity of a reduction (`access` is OPS_INC, OPS_MIN or OPS_MAX), as `bytes` of an element of
/// kind `kind` (EK_F32, EK_F64, EK_I32 or EK_I64).
bool reductionIdentity(int kind, int access, void *out8);

/// Host buffers (sequential and OpenMP backends). `parallel` allows OpenMP threads.
bool reductionFillHost(void *buffer, std::size_t n, int kind, int access, bool parallel);
bool reductionFoldHost(const void *buffer, std::size_t n, int kind, int access, bool parallel,
                       void *out8);

/// Combines `value` into `*dst`: `*dst = op(*dst, value)`.
bool reductionCombine(void *dst, const void *value, int kind, int access);

#ifdef OPS_ENABLE_CUDA
/// Device buffers. Kernels run on `stream`; `scratch` is a device buffer of at least
/// reductionScratchBytes() bytes that the fold uses for partial results and its answer.
std::size_t reductionScratchBytes();
bool reductionFillCuda(CUdeviceptr buffer, std::size_t n, int kind, int access, CUstream stream);
bool reductionFoldCuda(CUdeviceptr buffer, std::size_t n, int kind, int access, CUstream stream,
                       CUdeviceptr scratch, void *out8);
#endif

} // namespace ops_mlir

#endif // OPS_MLIR_RUNTIME_REDUCTION_H
