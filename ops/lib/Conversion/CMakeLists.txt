add_mlir_library(OPSMLIRConversion
  OpsToSCF.cpp
  MarkNoalias.cpp
  HoistAccumulators.cpp

  DEPENDS
  OPSMLIRConversionIncGen

  LINK_LIBS PUBLIC
  OPSMLIRDialect
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTransformUtils
  MLIRLLVMDialect
  MLIROpenMPDialect
  MLIRGPUDialect
)
