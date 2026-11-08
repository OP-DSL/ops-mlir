#include "ops-mlir/Dialect/OPS/IR/OPSDialect.h"
#include "ops-mlir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace ops_mlir::ops {

#define GEN_PASS_DEF_OPSPLAN
#include "ops-mlir/Transforms/Passes.h.inc"

namespace {

using namespace mlir;

/// The plan stage as a nested pipeline: the steps stay separate passes (each with its own tests, and visible to the
/// instrumentation of the parent pass manager), the options are those of this pass. The options of ops-fuse-point-local
/// are checked by that pass.
struct OpsPlanPass : public impl::OpsPlanBase<OpsPlanPass> {
  using OpsPlanBase::OpsPlanBase;

  LogicalResult initialize(MLIRContext *) override {
    FusePointLocalOptions fuse;
    fuse.maxLoops = maxLoops;
    fuse.reorder = reorder;
    fuse.placement = placement;
    fuse.guarded = guarded;
    fuse.boxRatio = boxRatio;
    plan = OpPassManager(ModuleOp::getOperationName());
    plan.addPass(createFusePointLocal(fuse));
    plan.addPass(createForwardStores());
    plan.addPass(createCanonicalizerPass());
    plan.addPass(createCSEPass());
    plan.addPass(createOutlineGroups());
    return success();
  }

  void runOnOperation() override {
    if (failed(runPipeline(plan, getOperation()))) signalPassFailure();
  }

  OpPassManager plan{ModuleOp::getOperationName()};
};

} // namespace

} // namespace ops_mlir::ops
