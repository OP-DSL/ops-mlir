#!/bin/bash -l
#SBATCH --job-name=opsjit-cir
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.log
# P5.1: clang with ClangIR (-fclangir -emit-cir) from the same LLVM 23.1.0 sources as the pinned toolchain. Only clang and
# MLIR are built (no lld, runtimes, NVPTX), into ~/ops-jit-cl/build/llvm-cir; nothing is installed. Use:
#   sbatch -D $PWD -o $HOME/ops-jit-cl/logs/%x-%j.log cluster/v2/build_cir.sbatch
set -euo pipefail
source /opt/rh/gcc-toolset-13/enable
SRC=$HOME/MLIR/llvm-project-23.1.0.src/llvm
BUILD=${CIR_BUILD:-$HOME/ops-jit-cl/build/llvm-cir}
mkdir -p "$BUILD"
echo "== $(date -Is) configure $SRC -> $BUILD"
cmake -G Ninja -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(command -v gcc)" -DCMAKE_CXX_COMPILER="$(command -v g++)" \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DCLANG_ENABLE_CIR=ON \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DMLIR_INCLUDE_INTEGRATION_TESTS=OFF \
  -DLLVM_ENABLE_Z3_SOLVER=OFF -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_LIBXML2=OFF \
  -DCLANG_ENABLE_STATIC_ANALYZER=OFF -DCLANG_ENABLE_ARCMT=OFF \
  -DLLVM_USE_LINKER=gold \
  -DLLVM_PARALLEL_COMPILE_JOBS=32 -DLLVM_PARALLEL_LINK_JOBS=8
echo "== $(date -Is) build clang, cir-opt"
cmake --build "$BUILD" -j32 --target clang
cmake --build "$BUILD" -j32 --target cir-opt || echo "no cir-opt target"
echo "== $(date -Is) done"
"$BUILD/bin/clang" --version
