# End-to-end tests: OPS loops -> JIT -> host-reference comparison, built once
# per precision. fp32 is the precision the target GPU (RTX 4050) is good at;
# fp64 runs on the CPU backends by default.
#
# Requires the `ops_seq_local` OPS sequential library defined by the app
# CMake files (apps/c/laplace_2d), so `apps` must be added first.

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(mlir_extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(OPS_TEST_CUDA_RUNTIME "${MLIR_BUILD_DIR}/lib/libmlir_cuda_runtime.so")

# add_ops_e2e(<target> <source> <f32|f64>)
function(add_ops_e2e target source prec)
  string(TOUPPER ${prec} PREC_UP)
  add_executable(${target} ${source})
  target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include ${OPS_ROOT}/include)
  target_compile_definitions(${target} PRIVATE
    OPS_TEST_${PREC_UP}
    OPS_TEST_KERNEL_HEADER="${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernels_${prec}.h")
  target_link_libraries(${target} PRIVATE
    OPSRuntime MLIROPSDialect
    ${mlir_dialect_libs} ${mlir_conversion_libs} ${mlir_extension_libs}
    MLIRRegisterAllPasses MLIRIR MLIRPass MLIRSupport
    ops_seq_local m)
  target_link_options(${target} PRIVATE -no-pie)
endfunction()

# add_ops_e2e_test(<target> <backend> [SUFFIX s] [LABELS ...] [ENV k=v ...])
function(add_ops_e2e_test target backend)
  cmake_parse_arguments(A "" "SUFFIX" "LABELS;ENV" ${ARGN})
  set(name ${target}_${backend}${A_SUFFIX})
  add_test(NAME ${name} COMMAND ${target})
  set_tests_properties(${name} PROPERTIES
    ENVIRONMENT "OPS_BACKEND=${backend};OPS_MLIR_CUDA_RUNTIME=${OPS_TEST_CUDA_RUNTIME};${A_ENV}"
    LABELS "${A_LABELS}"
    FAIL_REGULAR_EXPRESSION "\\[FAIL\\]")
endfunction()

foreach(prec f64 f32)
  add_ops_e2e(fusion_cases_${prec} e2e/fusion_cases.cpp ${prec})
  add_ops_e2e_test(fusion_cases_${prec} seq LABELS cpu)
  add_ops_e2e_test(fusion_cases_${prec} openmp LABELS cpu)
  # Fusion modes: off (every loop alone) and identical-range-only.
  add_ops_e2e_test(fusion_cases_${prec} seq SUFFIX _nofusion LABELS cpu
    ENV OPS_MLIR_FUSION=0)
  add_ops_e2e_test(fusion_cases_${prec} seq SUFFIX _sameRange LABELS cpu
    ENV OPS_MLIR_FUSION_GUARDED=0)
  # Adjacent-only grouping (no dependence DAG), and latest-group placement.
  add_ops_e2e_test(fusion_cases_${prec} seq SUFFIX _noReorder LABELS cpu
    ENV OPS_MLIR_FUSION_REORDER=0)
  add_ops_e2e_test(fusion_cases_${prec} seq SUFFIX _latest LABELS cpu
    ENV OPS_MLIR_FUSION_PLACEMENT=latest)
endforeach()

if(OPS_ENABLE_CUDA)
  # fp32 is the supported GPU precision.
  add_ops_e2e_test(fusion_cases_f32 cuda LABELS gpu)
  add_ops_e2e_test(fusion_cases_f32 cuda SUFFIX _nofusion LABELS gpu
    ENV OPS_MLIR_FUSION=0)
  add_ops_e2e_test(fusion_cases_f32 cuda SUFFIX _sameRange LABELS gpu
    ENV OPS_MLIR_FUSION_GUARDED=0)
  add_ops_e2e_test(fusion_cases_f32 cuda SUFFIX _noReorder LABELS gpu
    ENV OPS_MLIR_FUSION_REORDER=0)
  # fp64 on this GPU is correctness-only; opt in with -L gpu-fp64.
  add_ops_e2e_test(fusion_cases_f64 cuda LABELS gpu-fp64)
endif()
