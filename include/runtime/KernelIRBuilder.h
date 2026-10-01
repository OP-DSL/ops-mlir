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
  int dim = 1;      // Gbl: number of elements
  mlir::Type elt;   // element type (f32/f64/i32); null if unsupported
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
  /// the kernel uses something unsupported.
  mlir::func::FuncOp generateAccessor(
      const std::vector<std::string> &sourceFiles, const std::string &kernelName,
      const std::string &functionName, int indexRank,
      const std::vector<KernelArgInfo> &args,
      const std::map<std::string, const void *> &constants, llvm::raw_ostream &errs);

private:
  mlir::MLIRContext &context_;
};

} // namespace ops_mlir

#endif // OPS_MLIR_RUNTIME_KERNEL_IR_BUILDER_H
