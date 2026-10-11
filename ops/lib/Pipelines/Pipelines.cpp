#include "ops-mlir/Pipelines/Pipelines.h"

#include "ops-mlir/Conversion/Passes.h"
#include "ops-mlir/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Errc.h"

#include <algorithm>

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
  if (o.getBool("fma", false)) pm.addPass(createOpsFpContract());
  // Arithmetic on half precision values is done in single precision (rounded back after every operation): the CPU has
  // no half precision arithmetic, only conversions.
  mlir::arith::ArithEmulateUnsupportedFloatsOptions half;
  half.sourceTypeStrs = {"f16"};
  half.targetTypeStr = "f32";
  pm.addPass(mlir::arith::createArithEmulateUnsupportedFloats(half));
  if (omp) pm.addPass(mlir::createConvertSCFToOpenMPPass());
  // The accumulators of reductions go to the entry of the parallel region (or the function). The OpenMP conversion
  // wraps the loop body in a memref.alloca_scope: with nothing left to allocate, canonicalisation removes it (a scope
  // whose body is lowered to several blocks, inner loops, would be invalid).
  pm.addPass(createOpsHoistAccumulators());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  // expand-strided-metadata runs its patterns greedily, canonicalisation of branches included: after the control flow
  // is in the LLVM dialect that merges the two identical successors of a `cond_br` (the loads of a conditional
  // expression) into one block with memref arguments, which the LLVM dialect cannot have. Before it, they are cf ops.
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(createOpsMarkNoalias());
  if (omp) pm.addPass(mlir::createConvertOpenMPToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  return llvm::Error::success();
}

/// Lowers a group for NVIDIA GPUs: one thread per point (convert-ops-to-scf target=gpu), the kernel to NVVM and a
/// fatbin through MLIR's gpu-lower-to-nvvm-pipeline (ptxas from the CUDA toolkit: CUDA_ROOT), the host side with
/// calls to the mgpu* runtime functions (opdsl GpuShim).
llvm::Error buildCUDA(mlir::OpPassManager &pm, const OptionMap &o) {
  OpsSpecializeOptions specialize;
  specialize.mode = o.getString("specialize", "none");
  if (specialize.mode != "none" && specialize.mode != "ranges" && specialize.mode != "shapes")
    return pipelineError("option 'specialize' must be none, ranges or shapes, got '" + specialize.mode + "'");
  // The pipeline option separates the block sizes with `x` (the option syntax uses commas to separate options).
  std::string block = o.getString("block", "32x4x1");
  std::replace(block.begin(), block.end(), 'x', ',');
  ConvertOpsToSCFOptions lower;
  lower.soa = o.getBool("soa", false);
  lower.target = "gpu";
  lower.block = block;

  pm.addPass(createOpsSpecialize(specialize));
  pm.addPass(createConvertOpsToSCF(lower));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (o.getBool("fma", false)) pm.addPass(createOpsFpContract());

  mlir::gpu::GPUToNVVMPipelineOptions nvvm;
  nvvm.cubinChip = o.getString("chip", "sm_80");
  nvvm.cubinFeatures = o.getString("features", "");
  std::string ptxas = o.getString("ptxas", "");
  std::replace(ptxas.begin(), ptxas.end(), ':', ' '); // options are separated by ":" in the option string
  nvvm.cmdOptions = ptxas;
  nvvm.optLevel = 3;
  // The result is a module in the LLVM dialect with a gpu.binary and gpu.launch_func: the translation to LLVM IR
  // (JITSession) turns the launches into calls of the mgpu* runtime functions (opdsl GpuShim).
  // The stages of mlir::gpu::buildLowerToNVVMPassPipeline (GPUToNVVMPipeline.cpp), with ops-gpu-kernel-attrs between the
  // conversion of the kernels and the serialisation.
  pm.addPass(mlir::createConvertNVGPUToNVVMPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addPass(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createConvertNVVMToLLVMPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  mlir::GpuNVVMAttachTargetOptions target;
  target.triple = nvvm.cubinTriple;
  target.chip = nvvm.cubinChip;
  target.features = nvvm.cubinFeatures;
  target.optLevel = nvvm.optLevel;
  target.cmdOptions = nvvm.cmdOptions;
  pm.addPass(mlir::createGpuNVVMAttachTarget(target));
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  mlir::ConvertIndexToLLVMPassOptions index;
  index.indexBitwidth = nvvm.indexBitWidth;
  pm.addPass(mlir::createConvertIndexToLLVMPass(index));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  mlir::ConvertGpuOpsToNVVMOpsOptions kernels;
  kernels.useBarePtrCallConv = nvvm.kernelUseBarePtrCallConv;
  kernels.indexBitwidth = nvvm.indexBitWidth;
  kernels.allowPatternRollback = nvvm.allowPatternRollback;
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertGpuOpsToNVVMOps(kernels));
  if (o.getBool("restrict", true)) pm.addNestedPass<mlir::gpu::GPUModuleOp>(createOpsGpuKernelAttrs());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCSEPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createReconcileUnrealizedCastsPass());
  mlir::GpuToLLVMConversionPassOptions host;
  host.hostBarePtrCallConv = nvvm.hostUseBarePtrCallConv;
  host.kernelBarePtrCallConv = nvvm.kernelUseBarePtrCallConv;
  pm.addPass(mlir::createGpuToLLVMConversionPass(host));
  mlir::GpuModuleToBinaryPassOptions binary;
  binary.compilationTarget = nvvm.cubinFormat;
  pm.addPass(mlir::createGpuModuleToBinaryPass(binary));
  pm.addPass(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
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
                 {"soa", "bool", "structure-of-arrays layout of multi-component dats"},
                 {"fma", "bool", "allow fused multiply-adds (contract fast-math flags)"}},
                buildCPU});
  registry.add({"ops-jit-cuda",
                "lower a group to a CUDA kernel launch (host code in the LLVM dialect, kernel as a fatbin)",
                {{"chip", "string", "GPU architecture, e.g. sm_80"},
                 {"features", "string", "NVVM target features, e.g. +ptx80"},
                 {"block", "string", "thread block size, e.g. 32x4x1 (x fastest)"},
                 {"ptxas", "string", "options for ptxas"},
                 {"specialize", "string", "none, ranges or shapes"},
                 {"soa", "bool", "structure-of-arrays layout of multi-component dats"},
                 {"fma", "bool", "allow fused multiply-adds (contract fast-math flags)"},
                 {"restrict", "bool", "mark the dat pointers of the kernels noalias (default true)"}},
                buildCUDA});
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
