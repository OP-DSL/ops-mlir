#===============================================================================
# CloverLeaf 2D from the OPS repository, compiled unmodified.
#
# CloverLeaf is GPL-3 and this project is MIT, so its sources are not copied here:
# they are read from the OPS checkout (OPS_ROOT/../../apps/c/CloverLeaf) and the
# shim/ops_seq_v2.h in this directory routes their ops_par_loop through ops-mlir.
#===============================================================================

get_filename_component(CLOVER_DIR "${OPS_ROOT}/../../apps/c/CloverLeaf" ABSOLUTE)
if(NOT EXISTS "${CLOVER_DIR}/clover_leaf.cpp")
  message(STATUS "CloverLeaf 2D not found at ${CLOVER_DIR}; skipping cloverleaf_2d")
  return()
endif()

file(GLOB CLOVER_SRCS "${CLOVER_DIR}/*.cpp")
list(FILTER CLOVER_SRCS EXCLUDE REGEX "/profile\\.cpp$")   # not part of the dev build
file(GLOB CLOVER_KERNEL_HEADER_LIST "${CLOVER_DIR}/*_kernel.h")
string(JOIN ":" CLOVER_KERNEL_HEADERS ${CLOVER_KERNEL_HEADER_LIST})

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(mlir_extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_executable(cloverleaf_2d ${CLOVER_SRCS} main_wrapper.cpp)

# The original main() becomes clover_original_main; main_wrapper.cpp supplies main().
set_source_files_properties("${CLOVER_DIR}/clover_leaf.cpp"
  PROPERTIES COMPILE_DEFINITIONS "main=clover_original_main")
set_source_files_properties(main_wrapper.cpp
  PROPERTIES COMPILE_DEFINITIONS "CLOVER_KERNEL_HEADERS=\"${CLOVER_KERNEL_HEADERS}\"")

# The shim directory must come first so that "ops_seq_v2.h" resolves to it.
target_include_directories(cloverleaf_2d BEFORE PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/shim)
target_include_directories(cloverleaf_2d PRIVATE
  ${CLOVER_DIR} ${CMAKE_SOURCE_DIR}/include ${OPS_ROOT}/include)

target_link_libraries(cloverleaf_2d PRIVATE
  OPSRuntime MLIROPSDialect
  ${mlir_dialect_libs} ${mlir_conversion_libs} ${mlir_extension_libs}
  MLIRRegisterAllPasses MLIRIR MLIRPass MLIRSupport
  ops_seq_local m)
# -rdynamic: the runtime names kernels by their address (dladdr), see kernelSymbolName.
target_link_options(cloverleaf_2d PRIVATE -no-pie -rdynamic)
