add_mlir_library(OPSMLIRTransforms
  MaterializeKernels.cpp
  SpecializeConsts.cpp

  DEPENDS
  OPSMLIRTransformsIncGen

  LINK_LIBS PUBLIC
  OPSMLIRDialect
  OPSMLIRAnalysis
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRTransformUtils
  MLIRFunctionInterfaces
)
