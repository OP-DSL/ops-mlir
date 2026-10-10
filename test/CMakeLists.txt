find_package(Python3 REQUIRED COMPONENTS Interpreter)

# lit is vendored (third_party/lit): the pinned LLVM install ships FileCheck, not and count, but no lit.
set(OPSJIT_LIT_PY ${CMAKE_SOURCE_DIR}/third_party/lit/lit.py)
set(OPSJIT_LLVM_TOOLS_DIR ${LLVM_TOOLS_BINARY_DIR})

configure_file(lit/lit.site.cfg.py.in ${CMAKE_CURRENT_BINARY_DIR}/lit/lit.site.cfg.py @ONLY)

add_custom_target(check-ops-mlir-lit
  COMMAND ${Python3_EXECUTABLE} ${OPSJIT_LIT_PY} -sv ${CMAKE_CURRENT_BINARY_DIR}/lit
  DEPENDS opsjit-tools
  USES_TERMINAL)
add_dependencies(check-ops-mlir check-ops-mlir-lit)
