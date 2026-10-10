# Standalone developer utilities (not part of any app or test suite).

add_executable(kernel-ir-dump kernel-ir-dump.cpp)
target_link_libraries(kernel-ir-dump PRIVATE OPSRuntime LLVMSupport)
# This LLVM/MLIR build has LLVM_ENABLE_RTTI=OFF, so its static libraries
# contain no typeinfo for LLVM classes. kernel-ir-dump.cpp is the first
# target here to use llvm::cl::opt/cl::list, which need cl::Option's
# typeinfo -- compiling this TU with RTTI on (the GCC default) then leaves
# "undefined reference to typeinfo for llvm::cl::Option" at link time.
# Match the library's RTTI setting instead.
target_compile_options(kernel-ir-dump PRIVATE -fno-rtti)

# --- CTest coverage --------------------------------------------------------
# KernelIRBuilderTest.cpp already covers KernelIRBuilder itself against
# synthetic snippets; these exercise the *tool* end-to-end against real
# kernel sources shipped in this repo, so a regression in either the CLI
# plumbing or the real kernels' translatability shows up in `ctest`.

add_test(NAME KernelIRDump_SingleReturn
  COMMAND kernel-ir-dump
    ${CMAKE_SOURCE_DIR}/apps/c/laplace_2d/laplace_kernels.h
    apply_stencil 2)
set_tests_properties(KernelIRDump_SingleReturn PROPERTIES
  PASS_REGULAR_EXPRESSION "func.func private @apply_stencil.*-> f64")

add_test(NAME KernelIRDump_OutPointer
  COMMAND kernel-ir-dump
    ${CMAKE_SOURCE_DIR}/apps/c/taylor_green_vortex/opensbliblock00_kernels.h
    opensbliblock00Kernel039 3
    -double Delta0block0=0.1 -double Delta1block0=0.1 -double Delta2block0=0.1
    -double Minf=0.1 -double gama=1.4)
set_tests_properties(KernelIRDump_OutPointer PROPERTIES
  PASS_REGULAR_EXPRESSION "memref<3xi32>.*memref<5xf64>")

add_test(NAME KernelIRDump_UnknownKernelFails
  COMMAND kernel-ir-dump
    ${CMAKE_SOURCE_DIR}/apps/c/laplace_2d/laplace_kernels.h
    does_not_exist 2)
set_tests_properties(KernelIRDump_UnknownKernelFails PROPERTIES WILL_FAIL TRUE)

# Single-precision kernels translate to f32, with no f64 anywhere.
add_test(NAME KernelIRDump_Float
  COMMAND kernel-ir-dump
    ${CMAKE_SOURCE_DIR}/tests/kernels/kernels_f32.h
    k_lap5 2)
set_tests_properties(KernelIRDump_Float PROPERTIES
  PASS_REGULAR_EXPRESSION "func.func private @k_lap5.*-> f32"
  FAIL_REGULAR_EXPRESSION "f64")

add_test(NAME KernelIRDump_FloatOutPointer
  COMMAND kernel-ir-dump
    ${CMAKE_SOURCE_DIR}/tests/kernels/kernels_f32.h
    k_two 2)
set_tests_properties(KernelIRDump_FloatOutPointer PROPERTIES
  PASS_REGULAR_EXPRESSION "memref<2xf32>"
  FAIL_REGULAR_EXPRESSION "f64")
