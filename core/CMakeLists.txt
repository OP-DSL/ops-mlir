# opdsl: infrastructure shared by DSL-specific MLIR JITs (OPS, and later OP2): hashing, kernel artifacts,
# ORC JIT session, persistent compiled-code cache, GPU launch shim, pipeline registry, diagnostics.
#
# Rule: nothing in core/ may include an OPS header or mention OPS. scripts/check_core_independence.sh
# enforces it (ctest: core_independence). That keeps core/ extractable into a repository of its own.

add_library(OPDSLSupport STATIC
  lib/Support/Hash.cpp)
target_include_directories(OPDSLSupport PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(OPDSLSupport PUBLIC LLVMSupport)

add_library(OPDSLPipeline STATIC
  lib/Pipeline/PipelineRegistry.cpp)
target_include_directories(OPDSLPipeline PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(OPDSLPipeline PUBLIC LLVMSupport MLIRPass MLIRIR)

llvm_map_components_to_libnames(opdsl_llvm_libs Core Support OrcJIT Passes native TargetParser)
add_library(OPDSLJIT STATIC
  lib/JIT/JITSession.cpp)
target_include_directories(OPDSLJIT PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(OPDSLJIT PUBLIC
  MLIRIR MLIRExecutionEngine MLIRTargetLLVMIRExport MLIRBuiltinToLLVMIRTranslation MLIRLLVMToLLVMIRTranslation
  MLIROpenMPToLLVMIRTranslation ${opdsl_llvm_libs})

add_library(OPDSLCache STATIC
  lib/Cache/PersistentCache.cpp)
target_include_directories(OPDSLCache PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(OPDSLCache PUBLIC LLVMSupport)

add_library(OPDSLTestMain STATIC unittests/TestMain.cpp)
target_include_directories(OPDSLTestMain PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(unittests)

add_test(NAME core_independence
  COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/check_core_independence.sh ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(core_independence PROPERTIES LABELS "unit")
