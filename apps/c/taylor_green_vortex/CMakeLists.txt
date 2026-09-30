#===============================================================================
# Taylor Green Vortex Application

if(NOT DEFINED ENV{HDF5_INSTALL_PATH})
  message(FATAL_ERROR "Environment variable HDF5_INSTALL_PATH is not set. See README.md.")
endif()
set(HDF5_INSTALL_PATH $ENV{HDF5_INSTALL_PATH})

# Collect OPS sequential backend sources from the OPS source tree
file(GLOB OPS_CORE_SRCS   "${OPS_ROOT}/src/core/*.cpp")
file(GLOB OPS_SEQ_SRCS    "${OPS_ROOT}/src/sequential/*.cpp")
file(GLOB OPS_EXTERN_SRCS "${OPS_ROOT}/src/externlib/*.cpp")

# Exclude device-specific files that require CUDA/MPI (keep HDF5 sources - this app needs them)
list(FILTER OPS_CORE_SRCS   EXCLUDE REGEX "device")

add_library(ops_seq_hdf5_local STATIC
  ${OPS_CORE_SRCS}
  ${OPS_SEQ_SRCS}
  ${OPS_EXTERN_SRCS}
)
target_include_directories(ops_seq_hdf5_local PUBLIC
  ${OPS_ROOT}/include
  ${HDF5_INSTALL_PATH}/include
)
target_compile_options(ops_seq_hdf5_local PRIVATE -fno-PIE)

#===============================================================================

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(mlir_extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_executable(opensbli opensbli.cpp)

target_include_directories(opensbli
  PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${OPS_ROOT}/include
  ${HDF5_INSTALL_PATH}/include
)

target_link_directories(opensbli PRIVATE ${HDF5_INSTALL_PATH}/lib)

target_link_libraries(opensbli
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
  ops_seq_hdf5_local
  hdf5_hl
  hdf5
  m
)

target_link_options(opensbli PRIVATE -no-pie)

#===============================================================================
# Single-precision variant, generated from the sources above so it cannot
# drift. Literals, types, type-name strings and math calls are all converted
# (see to_single_precision.py); check_single_precision.py verifies it.

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(TGV_F32_DIR ${CMAKE_CURRENT_BINARY_DIR}/single)
set(TGV_FILES constants.h opensbliblock00_kernels.h defdec_data_set.h io.h
    opensbli.cpp bc_exchanges.h stencils.h diagnostics.h)
set(TGV_F32_OUT "")
set(TGV_SRC_DEPS "")
foreach(f ${TGV_FILES})
  list(APPEND TGV_F32_OUT ${TGV_F32_DIR}/${f})
  list(APPEND TGV_SRC_DEPS ${CMAKE_CURRENT_SOURCE_DIR}/${f})
endforeach()

add_custom_command(
  OUTPUT ${TGV_F32_OUT}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/to_single_precision.py
          ${CMAKE_CURRENT_SOURCE_DIR} ${TGV_F32_DIR}
  DEPENDS ${TGV_SRC_DEPS} ${CMAKE_CURRENT_SOURCE_DIR}/to_single_precision.py
  COMMENT "Generating single-precision Taylor-Green vortex sources")

add_executable(opensbli_f32 ${TGV_F32_DIR}/opensbli.cpp)
target_sources(opensbli_f32 PRIVATE ${TGV_F32_OUT})
target_include_directories(opensbli_f32 PRIVATE
  ${CMAKE_SOURCE_DIR}/include ${OPS_ROOT}/include ${HDF5_INSTALL_PATH}/include)
target_link_directories(opensbli_f32 PRIVATE ${HDF5_INSTALL_PATH}/lib)
target_link_libraries(opensbli_f32 PRIVATE
  OPSRuntime MLIROPSDialect
  ${mlir_dialect_libs} ${mlir_conversion_libs} ${mlir_extension_libs}
  MLIRRegisterAllPasses MLIRIR MLIRPass MLIRSupport
  ops_seq_hdf5_local hdf5_hl hdf5 m)
target_link_options(opensbli_f32 PRIVATE -no-pie)

# Static + JIT-level proof that nothing in the single-precision kernels
# computes in double.
add_test(NAME tgv_f32_no_upcast
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_single_precision.py
          ${CMAKE_CURRENT_SOURCE_DIR} ${TGV_F32_DIR} $<TARGET_FILE:kernel-ir-dump>)
set_tests_properties(tgv_f32_no_upcast PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")

# f32 vs f64 on a small grid: single precision must stay within single-
# precision distance of double (and run strict: no implicit promotion).
foreach(be seq openmp)
  add_test(NAME tgv_f32_vs_f64_${be}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_tgv_precision.py
            ${CMAKE_CURRENT_BINARY_DIR} ${be})
  set_tests_properties(tgv_f32_vs_f64_${be} PROPERTIES LABELS cpu
    FAIL_REGULAR_EXPRESSION "FAIL")
endforeach()
if(OPS_ENABLE_CUDA)
  add_test(NAME tgv_f32_vs_f64_cuda
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_tgv_precision.py
            ${CMAKE_CURRENT_BINARY_DIR} cuda)
  set_tests_properties(tgv_f32_vs_f64_cuda PROPERTIES LABELS gpu
    ENVIRONMENT "OPS_MLIR_CUDA_RUNTIME=${MLIR_BUILD_DIR}/lib/libmlir_cuda_runtime.so"
    FAIL_REGULAR_EXPRESSION "FAIL")
endif()
