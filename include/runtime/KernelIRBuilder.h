//===- KernelIRBuilder.h - C++ kernel body -> MLIR --------------*- C++ -*-===//
//
// Translates a single OPS kernel function's C++ source into a real MLIR
// function body, for backends (e.g. CUDA).
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_RUNTIME_KERNEL_IR_BUILDER_H
#define OPS_MLIR_RUNTIME_KERNEL_IR_BUILDER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <map>
#include <string>
#include <vector>

namespace ops_mlir {

/// What one ops_par_loop argument means to a kernel, for accessor-style kernels
/// (see generateAccessor). One per argument, in argument order.
struct KernelArgInfo {
  enum class Kind { Dat, Gbl, Idx, Reduce };
  Kind kind = Kind::Dat;
  int access = 0; // OPS_READ 0, WRITE 1, RW 2, INC 3, MIN 4, MAX 5
  /// Dat: the stencil's points, in the order they were declared and in OPS's
  /// dimension order (x first), padded with zeros to three dimensions.
  std::vector<std::array<int, 3>> points;
  bool scaled = false;  // Dat: ... and some stride is neither 0 nor 1 (multigrid)
  bool strided = false; // Dat: the stencil's stride is not 1 in every dimension
  int dim = 1;      // Gbl: number of elements
  mlir::Type elt;   // element type (f32/f64/i32); null if unsupported
};

/// A registered extern constant (or a member of one) that an accessor kernel reads.
/// The translated function takes it as a trailing scalar parameter, after the
/// global elements, in the order of the list generateAccessor returns; the runtime
/// reads the current value from `name`'s registered address plus `offset` when the
/// loop is enqueued, because values like the time step change while the program runs.
struct KernelConstRef {
  std::string name;
  int64_t offset = 0;
  mlir::Type elt;
};

/// What translating a kernel revealed about it, beyond the function itself.
struct KernelTraits {
  /// Assigns to an INC reduction (`*r = e`) instead of accumulating (`*r += e`).
  bool assignStyleReduction = false;
  /// Bit k is set when the kernel reads idx[k] of an ops_arg_idx.
  unsigned idxAxesRead = 0;
  /// Registered integer constants the generated code was specialised on (loop bounds, array
  /// subscripts), with the value it assumed.
  std::vector<std::pair<std::string, int64_t>> specialized;
};

class KernelIRBuilder {
public:
  explicit KernelIRBuilder(mlir::MLIRContext &context) : context_(context) {}

  mlir::func::FuncOp generate(const std::string &sourceFile,
                              const std::string &kernelName, int indexRank,
                              const std::map<std::string, const void *> &constants,
                              llvm::raw_ostream &errs);

  /// Translates a kernel written with ACC<T> accessors (any control flow the
  /// translator supports) into a function named `functionName` whose signature
  /// follows from `args` (see AccessorKernel.cpp). The kernel is searched for in
  /// every file of `sourceFiles`. Returns a null op, with the reason on `errs`, if
  /// the kernel uses something unsupported. `preamble` is source text placed before
  /// each file: the includes the application's own translation unit had in front of
  /// the kernel headers (kernel files often rely on them).
  mlir::func::FuncOp generateAccessor(
      const std::vector<std::string> &sourceFiles, const std::string &kernelName,
      const std::string &functionName, int indexRank,
      const std::vector<KernelArgInfo> &args,
      const std::map<std::string, const void *> &constants, llvm::raw_ostream &errs,
      std::vector<KernelConstRef> *constRefs = nullptr,
      const std::string &preamble = "", KernelTraits *traits = nullptr);

private:
  mlir::MLIRContext &context_;
};

} // namespace ops_mlir

#endif // OPS_MLIR_RUNTIME_KERNEL_IR_BUILDER_H
