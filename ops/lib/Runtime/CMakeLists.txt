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

# The sources are an object library so that the unit tests can link them too (the shared library hides its symbols).
add_library(OPSMLIRRuntimeObjects OBJECT
  Options.cpp
  Artifact.cpp
  Claim.cpp
  Lifter.cpp
  Binding.cpp
  GroupCompiler.cpp
  Executor.cpp
  Init.cpp)
set_target_properties(OPSMLIRRuntimeObjects PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(OPSMLIRRuntimeObjects PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_include_directories(OPSMLIRRuntimeObjects SYSTEM PUBLIC "${OPS_INSTALL}/include")
set(OPSJIT_RUNTIME_LIBS
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
  MLIRPass
  CACHE INTERNAL "libraries libopsjit is made of")
target_link_libraries(OPSMLIRRuntimeObjects PUBLIC ${OPSJIT_RUNTIME_LIBS})

add_library(opsjit SHARED $<TARGET_OBJECTS:OPSMLIRRuntimeObjects>)
target_link_libraries(opsjit PRIVATE ${OPSJIT_RUNTIME_LIBS})
set_target_properties(opsjit PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# Keep the statically linked MLIR/LLVM out of the dynamic symbol table: the application may link its own LLVM.
target_link_options(opsjit PRIVATE -Wl,--exclude-libs,ALL)
# The OPS functions it calls (ops_lazy_*) are provided by the OPS library the application links.
target_link_options(opsjit PRIVATE -Wl,-z,undefs)
