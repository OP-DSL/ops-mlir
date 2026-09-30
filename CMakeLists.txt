cmake_minimum_required(VERSION 3.20)
project(ops-mlir CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#===============================================================================
# MLIR/LLVM Setup
#===============================================================================

# Paths are read from environment variables. See README.md for setup
# instructions (source env_setup or export these yourself before running cmake).
if(NOT DEFINED ENV{MLIR_BUILD_DIR})
  message(FATAL_ERROR "Environment variable MLIR_BUILD_DIR is not set. See README.md.")
endif()
if(NOT DEFINED ENV{LLVM_BUILD_DIR})
  message(FATAL_ERROR "Environment variable LLVM_BUILD_DIR is not set. See README.md.")
endif()
if(NOT DEFINED ENV{LLVM_SOURCE_DIR})
  message(FATAL_ERROR "Environment variable LLVM_SOURCE_DIR is not set. See README.md.")
endif()
if(NOT DEFINED ENV{MLIR_SOURCE_DIR})
  message(FATAL_ERROR "Environment variable MLIR_SOURCE_DIR is not set. See README.md.")
endif()
if(NOT DEFINED ENV{OPS_ROOT})
  message(FATAL_ERROR "Environment variable OPS_ROOT is not set. See README.md.")
endif()

set(MLIR_BUILD_DIR $ENV{MLIR_BUILD_DIR})
set(LLVM_BUILD_DIR $ENV{LLVM_BUILD_DIR})
set(LLVM_SOURCE_DIR $ENV{LLVM_SOURCE_DIR})
set(MLIR_SOURCE_DIR $ENV{MLIR_SOURCE_DIR})
set(OPS_ROOT $ENV{OPS_ROOT})
set(OPS_INCLUDE_DIR ${OPS_ROOT}/include)

# Find MLIR and LLVM packages
find_package(MLIR REQUIRED CONFIG
  HINTS "${MLIR_BUILD_DIR}/lib/cmake/mlir"
)
find_package(LLVM REQUIRED CONFIG
  HINTS "${LLVM_BUILD_DIR}/lib/cmake/llvm"
)
find_package(Clang REQUIRED CONFIG
  HINTS "${LLVM_BUILD_DIR}/lib/cmake/clang"
)

# Include MLIR cmake utilities
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_BUILD_DIR}/cmake/modules")

include(AddLLVM)
include(AddMLIR)
include(TableGen)

#===============================================================================
# CUDA GPU Backend Configuration
#===============================================================================
option(OPS_ENABLE_CUDA "Build the NVIDIA CUDA/NVVM backend pipeline" OFF)

if(OPS_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

#===============================================================================
# OpenMP Setup
#===============================================================================
# omp.h and libomp.so. For an in-tree LLVM build they live in the OpenMP runtime
# build alongside it (derived from MLIR_BUILD_DIR, no extra variable needed).
# For an *installed* LLVM, point OPS_OMP_INCLUDE_DIR (the directory with omp.h,
# usually <prefix>/lib/clang/<ver>/include) and OPS_OMP_LIB_DIR (the directory
# with libomp.so, <prefix>/lib or <prefix>/lib/<triple>) at them.
if(DEFINED ENV{OPS_OMP_INCLUDE_DIR} AND DEFINED ENV{OPS_OMP_LIB_DIR})
  set(OMP_INCLUDE_DIR $ENV{OPS_OMP_INCLUDE_DIR})
  set(OMP_LIB_DIR $ENV{OPS_OMP_LIB_DIR})
else()
  cmake_path(GET MLIR_BUILD_DIR PARENT_PATH LLVM_PROJECT_DIR)
  set(OMP_INCLUDE_DIR "${LLVM_PROJECT_DIR}/openmp/build/runtime/src")
  set(OMP_LIB_DIR "${OMP_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${OMP_INCLUDE_DIR}/omp.h")
  message(FATAL_ERROR "omp.h not found at ${OMP_INCLUDE_DIR}. Either build the LLVM "
    "OpenMP runtime in-tree alongside MLIR_BUILD_DIR, or set OPS_OMP_INCLUDE_DIR and "
    "OPS_OMP_LIB_DIR for an installed LLVM.")
endif()

add_compile_options(-fopenmp)
add_link_options(-fopenmp)

#===============================================================================
# Project Structure
#===============================================================================
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OPS_INCLUDE_DIR})
include_directories(${OMP_INCLUDE_DIR})
link_directories(${OMP_LIB_DIR})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${MLIR_BUILD_DIR}/include)
include_directories(${LLVM_BUILD_DIR}/include)
include_directories(${MLIR_SOURCE_DIR}/include)
include_directories(${LLVM_SOURCE_DIR}/include)

# Before any add_subdirectory that registers tests (apps/ does too).
enable_testing()

add_subdirectory(lib)
add_subdirectory(apps)

add_subdirectory(tools)
add_subdirectory(unittests)
add_subdirectory(tests)

#===============================================================================
# Compilation Database for Language Server
#===============================================================================
# Automatically generated at ${CMAKE_BINARY_DIR}/compile_commands.json
