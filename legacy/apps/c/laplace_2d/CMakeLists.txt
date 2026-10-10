#===============================================================================
# Laplace 2D Application

# Collect OPS sequential backend sources from the OPS source tree
file(GLOB OPS_CORE_SRCS   "${OPS_ROOT}/src/core/*.cpp")
file(GLOB OPS_SEQ_SRCS    "${OPS_ROOT}/src/sequential/*.cpp")
file(GLOB OPS_EXTERN_SRCS "${OPS_ROOT}/src/externlib/*.cpp")

# Exclude device-specific and HDF5 files that require CUDA/MPI/HDF5
list(FILTER OPS_CORE_SRCS   EXCLUDE REGEX "device")
list(FILTER OPS_EXTERN_SRCS EXCLUDE REGEX "hdf5")

add_library(ops_seq_local STATIC
  ${OPS_CORE_SRCS}
  ${OPS_SEQ_SRCS}
  ${OPS_EXTERN_SRCS}
)
target_include_directories(ops_seq_local PUBLIC ${OPS_ROOT}/include)
target_compile_options(ops_seq_local PRIVATE -fno-PIE)

#===============================================================================

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(mlir_extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_executable(laplace_2d laplace_2d.cpp)

target_include_directories(laplace_2d
  PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${OPS_ROOT}/include
)

target_link_libraries(laplace_2d
  PRIVATE
  OPSRuntime
  MLIROPSDialect
  ${mlir_dialect_libs}
  ${mlir_conversion_libs}
  ${mlir_extension_libs}
  MLIRRegisterAllPasses
  MLIRIR
  MLIRPass
  MLIRSupport
  ops_seq_local
  m
)

target_link_options(laplace_2d PRIVATE -no-pie)
