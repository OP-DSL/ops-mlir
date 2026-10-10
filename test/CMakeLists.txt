find_package(Python3 REQUIRED COMPONENTS Interpreter)

# lit is vendored (third_party/lit): the pinned LLVM install ships FileCheck, not and count, but no lit.
set(OPSJIT_LIT_PY ${CMAKE_SOURCE_DIR}/third_party/lit/lit.py)
set(OPSJIT_LLVM_TOOLS_DIR ${LLVM_TOOLS_BINARY_DIR})
set(OPSJIT_LLVM_LIBRARY_DIR ${LLVM_LIBRARY_DIR})
# The importer tests parse real OPS kernels: they need the headers of an OPS source tree (OPS_SRC).
set(OPSJIT_OPS_SRC "$ENV{OPS_SRC}" CACHE PATH "OPS source tree (the importer tests use ops/c/include)")
set(OPSJIT_OPS_INCLUDE_DIR "${OPSJIT_OPS_SRC}/ops/c/include")

configure_file(lit/lit.site.cfg.py.in ${CMAKE_CURRENT_BINARY_DIR}/lit/lit.site.cfg.py @ONLY)

add_custom_target(check-ops-mlir-lit
  COMMAND ${Python3_EXECUTABLE} ${OPSJIT_LIT_PY} -sv ${CMAKE_CURRENT_BINARY_DIR}/lit
  DEPENDS opsjit-tools
  USES_TERMINAL)
add_dependencies(check-ops-mlir check-ops-mlir-lit)
