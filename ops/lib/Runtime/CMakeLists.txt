# libopsjit: the runtime that plugs into OPS's lazy execution. It uses the OPS headers (OPS_INSTALL/include) and the
# C API of ops_lazy_executor.h, which the OPS backend library of the application provides at link time: this library
# is the same for the sequential, OpenMP, MPI and (later) device builds of an application.
if(NOT OPSJIT_BUILD_RUNTIME)
  return()
endif()
if(NOT OPS_INSTALL OR NOT EXISTS "${OPS_INSTALL}/include/ops_lazy_executor.h")
  message(STATUS "libopsjit not built: OPS_INSTALL='${OPS_INSTALL}' has no include/ops_lazy_executor.h (build OPS from feature/jit-executor with scripts/build_ops.sh)")
  return()
endif()

add_library(opsjit SHARED
  Options.cpp
  Artifact.cpp
  Claim.cpp
  Lifter.cpp
  Binding.cpp
  Executor.cpp
  Init.cpp)
target_include_directories(opsjit PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>")
target_include_directories(opsjit PRIVATE "${OPS_INSTALL}/include")
target_link_libraries(opsjit PRIVATE
  OPSMLIRDialect
  OPSMLIRAnalysis
  OPSMLIRTransforms
  OPSMLIRConversion
  OPSMLIRPipelines
  OPDSLSupport
  OPDSLPipeline
  OPDSLJIT
  OPDSLCache
  MLIRParser
  MLIRPass)
set_target_properties(opsjit PROPERTIES POSITION_INDEPENDENT_CODE ON LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# Keep the statically linked MLIR/LLVM out of the dynamic symbol table: the application may link its own LLVM.
target_link_options(opsjit PRIVATE -Wl,--exclude-libs,ALL)
# The OPS functions it calls (ops_lazy_*) are provided by the OPS library the application links.
target_link_options(opsjit PRIVATE -Wl,-z,undefs)
