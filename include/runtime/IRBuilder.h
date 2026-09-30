#ifndef OPS_MLIR_OPS_FRONTEND_OPS_BUILDER_H
#define OPS_MLIR_OPS_FRONTEND_OPS_BUILDER_H

#include "Dialect/OPS/OPSOps.h"
#include "runtime/Core.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <vector>

namespace ops_mlir {

/// Builder that converts captured OPS loops into an MLIR module.
class IRBuilder {
public:
  /// Create a builder with the given MLIR context.
  explicit IRBuilder(mlir::MLIRContext *ctx);

  /// Build an MLIR module from a queue of captured loop descriptions.
  /// Returns the generated module, or nullptr on error.
  /// `groupIds`, when given, has one fusion-group id per loop; it is attached
  /// as the `fuse_group` attribute so the lowering can merge those loops.
  mlir::ModuleOp buildModule(const std::vector<LoopDesc> &loops,
                             const std::vector<int64_t> *groupIds = nullptr);

  /// Print the module to a string for debugging.
  std::string moduleToString(mlir::ModuleOp module);

private:
  mlir::MLIRContext *ctx_;

  /// Build a single ops.par_loop operation from a LoopDesc.
  mlir::Operation *buildParLoopOp(const LoopDesc &loop,
                                  const int64_t *groupId = nullptr);

  /// Convert a LoopDesc argument into an OPS argument attribute.
  mlir::Attribute buildArgAttr(const ArgDesc &arg);

  ops_mlir::ops::DatAttr buildDatAttr(const DatDesc &dat);

  ops_mlir::ops::StencilAttr buildStencilAttr(const StencilDesc &stencil);
};

} // namespace ops_mlir

#endif // OPS_MLIR_OPS_FRONTEND_OPS_BUILDER_H
