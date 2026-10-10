add_mlir_library(OPSMLIRPipelines
  Pipelines.cpp

  LINK_LIBS PUBLIC
  OPDSLPipeline
  OPSMLIRTransforms
  OPSMLIRConversion
  MLIRPass
  MLIRTransforms
  MLIRSCFToOpenMP
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRArithToLLVM
  MLIRMathToLLVM
  MLIRFuncToLLVM
  MLIRMemRefTransforms
  MLIRMemRefToLLVM
  MLIROpenMPToLLVM
  MLIRReconcileUnrealizedCasts
)
