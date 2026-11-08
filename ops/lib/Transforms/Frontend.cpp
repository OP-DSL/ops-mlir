#include "ops-mlir/Dialect/OPS/IR/OPSDialect.h"
#include "ops-mlir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace ops_mlir::ops {

#define GEN_PASS_DEF_OPSFRONTEND
#include "ops-mlir/Transforms/Passes.h.inc"

namespace {

using namespace mlir;

/// The frontend stage as a nested pipeline: the steps stay separate passes (each with its own tests, and visible to the
/// instrumentation of the parent pass manager).
struct OpsFrontendPass : public impl::OpsFrontendBase<OpsFrontendPass> {
  using OpsFrontendBase::OpsFrontendBase;

  LogicalResult initialize(MLIRContext *) override {
    frontend = OpPassManager(ModuleOp::getOperationName());
    frontend.addPass(createMaterializeKernels());
    frontend.addPass(createSpecializeConsts());
    frontend.addPass(createCanonicalizerPass());
    frontend.addPass(createCSEPass());
    return success();
  }

  void runOnOperation() override {
    if (failed(runPipeline(frontend, getOperation()))) signalPassFailure();
  }

  OpPassManager frontend{ModuleOp::getOperationName()};
};

} // namespace

} // namespace ops_mlir::ops
