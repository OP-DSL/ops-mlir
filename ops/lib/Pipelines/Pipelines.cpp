#include "ops-mlir/Pipelines/Pipelines.h"

#include "ops-mlir/Conversion/Passes.h"
#include "ops-mlir/Transforms/Passes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Errc.h"

namespace ops_mlir::ops {

using opdsl::OptionMap;
using opdsl::PipelineRegistry;

namespace {

llvm::Error pipelineError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::errc::invalid_argument, message);
}

llvm::Error buildFrontend(mlir::OpPassManager &pm, const OptionMap &) {
  pm.addPass(createMaterializeKernels());
  pm.addPass(createSpecializeConsts());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  return llvm::Error::success();
}

llvm::Error buildPlan(mlir::OpPassManager &pm, const OptionMap &o) {
  FusePointLocalOptions fuse;
  fuse.maxLoops = o.getInt("max-loops", 8);
  fuse.reorder = o.getBool("reorder", true);
  fuse.placement = o.getString("placement", "earliest");
  if (fuse.placement != "earliest" && fuse.placement != "latest")
    return pipelineError("option 'placement' must be earliest or latest, got '" + fuse.placement + "'");
  fuse.guarded = o.getBool("guarded", false);
  pm.addPass(createFusePointLocal(fuse));
  pm.addPass(createForwardStores());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(createOutlineGroups());
  return llvm::Error::success();
}

llvm::Error buildCPU(mlir::OpPassManager &pm, const OptionMap &o) {
  OpsSpecializeOptions specialize;
  specialize.mode = o.getString("specialize", "none");
  if (specialize.mode != "none" && specialize.mode != "ranges" && specialize.mode != "shapes")
    return pipelineError("option 'specialize' must be none, ranges or shapes, got '" + specialize.mode + "'");
  ConvertOpsToSCFOptions lower;
  lower.soa = o.getBool("soa", false);
  bool omp = o.getBool("omp", true);

  pm.addPass(createOpsSpecialize(specialize));
  pm.addPass(createConvertOpsToSCF(lower));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (omp) {
    pm.addPass(mlir::createConvertSCFToOpenMPPass());
    // The conversion wraps the loop body in a memref.alloca_scope; an empty one is removed here, because a scope whose
    // body is lowered to several blocks (inner loops) is invalid.
    pm.addPass(mlir::createCanonicalizerPass());
  }
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(createOpsMarkNoalias());
  if (omp) pm.addPass(mlir::createConvertOpenMPToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  return llvm::Error::success();
}

} // namespace

void registerOPSPipelines(PipelineRegistry &registry) {
  registry.add({"ops-jit-frontend",
                "materialise kernels into loops, fold bound constants",
                {},
                buildFrontend});
  registry.add({"ops-jit-plan",
                "fuse loops, forward stores, outline the groups",
                {{"max-loops", "int", "most loops in a group"},
                 {"reorder", "bool", "let independent loops move past each other"},
                 {"placement", "string", "earliest or latest"},
                 {"guarded", "bool", "fuse loops of different ranges (not implemented)"}},
                buildPlan});
  registry.add({"ops-jit-cpu",
                "lower a group to the LLVM dialect for the host CPU",
                {{"omp", "bool", "parallelise with OpenMP"},
                 {"specialize", "string", "none, ranges or shapes"},
                 {"soa", "bool", "structure-of-arrays layout of multi-component dats"}},
                buildCPU});
}

void registerOPSPipelinesWithMLIR() {
  for (const PipelineRegistry::Entry *entry : PipelineRegistry::global().entries()) {
    std::string name = entry->name;
    mlir::registerPassPipeline(
        name, entry->description,
        [name](mlir::OpPassManager &pm, llvm::StringRef options,
               llvm::function_ref<mlir::LogicalResult(const llvm::Twine &)> reportError) {
          if (llvm::Error e = PipelineRegistry::global().build(name, options, pm))
            return reportError(llvm::toString(std::move(e)));
          return mlir::success();
        },
        [](llvm::function_ref<void(const mlir::detail::PassOptions &)>) {});
  }
}

} // namespace ops_mlir::ops
