# Builds an OPS application with the JIT:
#
#   ops_jit_add_app(<name> SOURCE_DIR <dir>
#                   [ODD <regex>...]       sources that other sources #include (no code generation, no compilation)
#                   [OTHERS <regex>...]    sources that need no code generation
#                   [EXTRA <regex>...]     sources the helper must ignore
#                   [HDF5]                 link the HDF5 builds of the OPS libraries
#                   [NO_SEQ_TEST]          no ctest for the sequential build (too slow: CloverLeaf, like in OPS)
#                   [TRANSLATOR_FLAGS <flags>...] [IMPORTER_FLAGS <flags>...]
#                   [TEST_SCRIPT <cmake_test.sh>])
#
# The helper translates the application with `ops-translator --jit` at configure time (like OPS's own
# build_ops_c_sample), imports its kernels (ops_add_jit_artifacts) and builds, for what the OPS installation and the
# toolchain offer,
#
#   <name>_jit            sequential/OpenMP     OPS::ops_seq
#   <name>_jit_mpi        MPI                   OPS::ops_mpi
#   <name>_jit_cuda       CUDA                  OPS::ops_cuda
#   <name>_jit_mpi_cuda   MPI + CUDA            OPS::ops_mpi_cuda
#
# all with -DOPS_LAZY, linked with libopsjit. With TEST_SCRIPT (OPS's apps/c/<app>/cmake_test.sh) and OPS_TEST=ON the
# same tests as the AOT builds run.
#
# Expects: OPS_INSTALL (find_package(OPS) was done), OPS_PYTHON and OPS_TRANSLATOR, the targets `opsjit` and the
# variable OPSJIT_IMPORTER.

include("${CMAKE_CURRENT_LIST_DIR}/OpsJitArtifacts.cmake")

function(_ops_jit_add_test exe_target script mode tmp)
  if(NOT OPS_TEST OR NOT EXISTS "${script}")
    return()
  endif()
  # The script prints the command line for the executable that the build will produce.
  execute_process(COMMAND /bin/bash ${script} "${CMAKE_CURRENT_BINARY_DIR}/${exe_target}" ${mode}
                  OUTPUT_VARIABLE command RESULT_VARIABLE rc OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT rc EQUAL 0 OR "${command}" STREQUAL "")
    return()
  endif()
  add_test(NAME ${exe_target}
           COMMAND ${CMAKE_COMMAND} "-DCMD=${command}" "-DARG= " -DOPS_INSTALL_PATH=${OPS_INSTALL}
                   -P ${OPS_SRC}/apps/c/runtests.cmake
           WORKING_DIRECTORY "${tmp}")
  set_tests_properties(${exe_target} PROPERTIES LABELS "jit-app")
endfunction()

function(ops_jit_add_app name)
  cmake_parse_arguments(ARG "HDF5;NO_SEQ_TEST" "SOURCE_DIR;TEST_SCRIPT" "ODD;OTHERS;EXTRA;TRANSLATOR_FLAGS;IMPORTER_FLAGS" ${ARGN})
  set(tmp "${CMAKE_CURRENT_BINARY_DIR}/${name}_tmp")
  file(REMOVE_RECURSE "${tmp}")
  file(MAKE_DIRECTORY "${tmp}")

  # Same preparation as OPS's build_ops_c_sample: copy the sources, run the translator on the ones that have loops.
  file(GLOB cpp "${ARG_SOURCE_DIR}/*.cpp")
  list(FILTER cpp EXCLUDE REGEX "ops.cpp")
  file(COPY ${cpp} DESTINATION "${tmp}")
  file(GLOB heads "${ARG_SOURCE_DIR}/*.h")
  file(COPY ${heads} DESTINATION "${tmp}")
  file(GLOB inputs "${ARG_SOURCE_DIR}/*.in")
  if(inputs)
    file(COPY ${inputs} DESTINATION "${tmp}")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${cpp} ${heads})

  file(GLOB dev "${tmp}/*.cpp")
  if(ARG_ODD)
    list(FILTER dev EXCLUDE REGEX "${ARG_ODD}")
  endif()
  foreach(pattern ${ARG_OTHERS} ${ARG_EXTRA})
    list(FILTER dev EXCLUDE REGEX "${pattern}")
  endforeach()
  set(names "")
  foreach(path ${dev})
    get_filename_component(file "${path}" NAME)
    list(APPEND names "${file}")
  endforeach()

  execute_process(
    COMMAND ${OPS_PYTHON} ${OPS_TRANSLATOR} -DOPS_ACC_IGNORE -I ${OPS_INSTALL}/include ${ARG_TRANSLATOR_FLAGS}
            --jit --file_paths ${names}
    WORKING_DIRECTORY "${tmp}" RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "ops-translator --jit failed for ${name} (${rc}):\n${out}\n${err}")
  endif()

  file(GLOB ops_sources "${tmp}/*ops*.cpp")
  if(ARG_ODD)
    list(FILTER ops_sources EXCLUDE REGEX "${ARG_ODD}")
  endif()
  file(GLOB others "${tmp}/*.cpp")
  list(REMOVE_ITEM others ${ops_sources} ${dev})
  if(ARG_ODD)
    list(FILTER others EXCLUDE REGEX "${ARG_ODD}")
  endif()
  foreach(pattern ${ARG_EXTRA})
    list(FILTER others EXCLUDE REGEX "${pattern}")
  endforeach()

  # The artifacts, imported once per application: they depend on the application's headers, not on the backend.
  add_library(${name}_jit_artifacts OBJECT)
  target_include_directories(${name}_jit_artifacts PRIVATE "${tmp}" "${OPS_INSTALL}/include")
  set_target_properties(${name}_jit_artifacts PROPERTIES POSITION_INDEPENDENT_CODE ON)
  ops_add_jit_artifacts(${name}_jit_artifacts KERNELS_JSON "${tmp}/jit_kernels.json"
                        IMPORTER ${OPSJIT_IMPORTER} CLANG_ARGS -DOPS_ACC_IGNORE ${ARG_IMPORTER_FLAGS})

  set(common_libs opsjit OpenMP::OpenMP_CXX)
  if(ARG_HDF5)
    set(hdf5_seq OPS::ops_hdf5_seq hdf5::hdf5 hdf5::hdf5_hl MPI::MPI_CXX)
    set(hdf5_mpi OPS::ops_hdf5_mpi hdf5::hdf5 hdf5::hdf5_hl MPI::MPI_CXX)
  endif()

  # <suffix> <kernels> <definitions> <libraries> <test mode>
  function(_variant suffix kernels defs libs mode)
    set(exe ${name}_jit${suffix})
    add_executable(${exe} ${ops_sources} ${others} "${tmp}/${kernels}" $<TARGET_OBJECTS:${name}_jit_artifacts>)
    target_include_directories(${exe} PRIVATE "${tmp}")
    target_compile_definitions(${exe} PRIVATE OPS_LAZY ${defs})
    target_link_libraries(${exe} PRIVATE ${libs} ${common_libs})
    if(NOT (ARG_NO_SEQ_TEST AND mode STREQUAL "seq"))
      _ops_jit_add_test(${exe} "${ARG_TEST_SCRIPT}" "${mode}" "${tmp}")
    endif()
  endfunction()

  _variant("" mpi_openmp/mpi_openmp_kernels.cpp "" "${hdf5_seq};OPS::ops_seq" seq)
  if(TARGET OPS::ops_mpi)
    _variant("_mpi" mpi_openmp/mpi_openmp_kernels.cpp "OPS_MPI" "${hdf5_mpi};OPS::ops_mpi;MPI::MPI_CXX" mpi)
  endif()
  if(OPSJIT_APPS_CUDA AND TARGET OPS::ops_cuda)
    _variant("_cuda" cuda/cuda_kernels.cu "" "${hdf5_seq};OPS::ops_cuda;CUDA::cudart_static;-lcurand;-lnvidia-ml" cuda)
    target_compile_options(${name}_jit_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    if(TARGET OPS::ops_mpi_cuda)
      _variant("_mpi_cuda" cuda/cuda_kernels.cu "OPS_MPI;MPICH_IGNORE_CXX_SEEK"
               "${hdf5_mpi};OPS::ops_mpi_cuda;MPI::MPI_CXX;CUDA::cudart_static;-lcurand;-lnvidia-ml" "mpi_cuda 1")
      target_compile_options(${name}_jit_mpi_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    endif()
  endif()
endfunction()
