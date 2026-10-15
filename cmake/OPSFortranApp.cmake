#===============================================================================
# OPSFortranApp.cmake - build helper for OPS-MLIR Fortran applications
#
# Part of OPS-MLIR Project. Distributed under the MIT License.
#
# An OPS Fortran application built through ops-mlir has two moving parts:
#
#   1. the OPS Fortran API modules (OPS_Fortran_Declarations, ...), compiled
#      from the OPS source tree;
#   2. the OPS-MLIR capture module, which replaces the translator-generated
#      OPS_Fortran_Reference and enqueues loops into the JIT.
#
# ops_add_fortran_application() wires both together. Unlike the OPS Fortran
# flow, no translator step runs: the application source is compiled as
# written, and the JIT translates the kernel module from source at run time,
# as it does a C++ application's kernel header.
#===============================================================================

if(NOT OPS_ENABLE_FORTRAN)
  return()
endif()

set(OPS_FORTRAN_SRC_DIR "${OPS_ROOT}/../fortran/src" CACHE PATH
  "Path to the OPS Fortran API sources (ops_for_declarations.F90, ...)")

# Applications link C++ libraries (the runtime, MLIR), so CMake links them
# with the C++ driver, which knows nothing of flang's runtime library. CMake
# would otherwise rely on detecting flang's implicit link libraries, which
# is not dependable, so ask flang where its runtime is and link it directly
# (see ops_ops_fortran_api_library).
execute_process(
  COMMAND "${OPS_FLANG_ROOT}/bin/flang" -print-file-name=libflang_rt.runtime.a
  OUTPUT_VARIABLE OPS_FLANG_RUNTIME_LIB
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT EXISTS "${OPS_FLANG_RUNTIME_LIB}")
  message(FATAL_ERROR
    "flang's runtime library (libflang_rt.runtime.a) was not found; the "
    "Fortran applications need it to link. Build it with\n"
    "  ninja runtimes/flang-rt\n"
    "in the LLVM build, or configure with -DOPS_BUILD_FORTRAN_APPS=OFF.")
endif()

#===============================================================================
# ops_seq_backend_library - the OPS sequential C backend, Fortran flavour
#===============================================================================
# Built from the same OPS sources as the C applications' ops_seq_local, but
# with OPS_FTN defined. That macro expands OPS_FTN_INTEROP to `extern "C"`,
# giving the OPS API C linkage so the ISO_C_BINDING interfaces in
# OPS_Fortran_Declarations can bind to it -- without it every `bind(C)`
# declaration fails to resolve against a C++-mangled symbol. OPS makes the
# same split, shipping `ops_seq` and `ops_for_seq` as separate libraries.
#===============================================================================
function(ops_seq_backend_library)
  if(TARGET ops_seq_local_ftn)
    return()
  endif()

  file(GLOB _core_srcs   "${OPS_ROOT}/src/core/*.cpp")
  file(GLOB _seq_srcs    "${OPS_ROOT}/src/sequential/*.cpp")
  file(GLOB _extern_srcs "${OPS_ROOT}/src/externlib/*.cpp")
  list(FILTER _core_srcs   EXCLUDE REGEX "device")
  list(FILTER _extern_srcs EXCLUDE REGEX "hdf5")

  add_library(ops_seq_local_ftn STATIC
    ${_core_srcs} ${_seq_srcs} ${_extern_srcs})
  target_include_directories(ops_seq_local_ftn PUBLIC ${OPS_ROOT}/include)
  target_compile_definitions(ops_seq_local_ftn PUBLIC OPS_FTN)
  target_compile_options(ops_seq_local_ftn PRIVATE -fno-PIE)
endfunction()

#===============================================================================
# ops_ops_fortran_api_library - compile the OPS Fortran API modules once
#===============================================================================
function(ops_ops_fortran_api_library)
  if(TARGET ops_fortran_api)
    return()
  endif()

  set(_sources
    "${OPS_FORTRAN_SRC_DIR}/ops_for_declarations.F90"
    "${OPS_FORTRAN_SRC_DIR}/ops_for_rt_support.F90")
  foreach(_source ${_sources})
    if(NOT EXISTS "${_source}")
      message(FATAL_ERROR
        "OPS Fortran source not found: ${_source}\n"
        "Set OPS_FORTRAN_SRC_DIR to the OPS checkout's ops/fortran/src.")
    endif()
  endforeach()

  ops_seq_backend_library()

  add_library(ops_fortran_api STATIC ${_sources})
  set_target_properties(ops_fortran_api PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/ops_fortran_modules")
  target_include_directories(ops_fortran_api PUBLIC
    "${OPS_ROOT}/../fortran/include")
  # These modules are thin ISO_C_BINDING wrappers over the OPS C core, so the
  # dependency has to be declared for static-library link ordering to work.
  target_link_libraries(ops_fortran_api PUBLIC ops_seq_local_ftn)
  # Every Fortran library and application links through this one, so the
  # runtime attached here lands after all Fortran objects on the link line.
  target_link_libraries(ops_fortran_api PUBLIC "${OPS_FLANG_RUNTIME_LIB}")
endfunction()

#===============================================================================
# ops_mlir_fortran_capture_library - the OPS_MLIR_Fortran_Reference module
#===============================================================================
function(ops_mlir_fortran_capture_library)
  if(TARGET ops_mlir_fortran_capture)
    return()
  endif()

  ops_ops_fortran_api_library()

  add_library(ops_mlir_fortran_capture STATIC
    "${CMAKE_SOURCE_DIR}/lib/runtime/fortran/ops_mlir_fortran_reference.F90")
  set_target_properties(ops_mlir_fortran_capture PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/ops_fortran_modules")
  target_include_directories(ops_mlir_fortran_capture PUBLIC
    "${CMAKE_CURRENT_BINARY_DIR}/ops_fortran_modules")
  target_link_libraries(ops_mlir_fortran_capture PUBLIC ops_fortran_api)
endfunction()

#===============================================================================
# ops_add_fortran_application(<target> <source>...)
#
# Builds an OPS Fortran application from its sources -- the driver, the
# kernel module and any modules they use. CMake works out the module
# compilation order itself. The kernel module is compiled in like any other
# source: the driver `use`s it and passes its procedures to ops_par_loop,
# even though the JIT translates the kernel bodies from source at run time.
#===============================================================================
function(ops_add_fortran_application target)
  if(NOT ARGN)
    message(FATAL_ERROR
      "ops_add_fortran_application(${target}) needs at least one source")
  endif()

  ops_mlir_fortran_capture_library()

  add_executable(${target} ${ARGN})

  set_target_properties(${target} PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${target}_modules")

  target_include_directories(${target} PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/ops_fortran_modules"
    "${OPS_ROOT}/include")

  get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
  get_property(mlir_extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

  target_link_libraries(${target} PRIVATE
    ops_mlir_fortran_capture
    OPSRuntimeFtn
    MLIROPSDialect
    ${mlir_dialect_libs}
    ${mlir_conversion_libs}
    ${mlir_extension_libs}
    MLIRRegisterAllPasses
    MLIRIR
    MLIRPass
    MLIRSupport
    ops_seq_local_ftn
    m)

  # The OPS C core and the JIT-compiled code are linked non-PIE, matching
  # the C++ applications.
  target_link_options(${target} PRIVATE -no-pie)
endfunction()
