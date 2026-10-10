#!/bin/bash -l
#SBATCH --job-name=opsjit-hello
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.log
# Toolchain smoke test (P0.0). Submit from a work tree on the cluster:
#
#   cd ~/ops-jit-cl/work/ops-mlir
#   J="-D $PWD -o $HOME/ops-jit-cl/logs/%x-%j.log"
#   sbatch $J cluster/v2/hello.sbatch                                    # CPU node
#   sbatch $J -p gpu --gres=gpu:a100:1 cluster/v2/hello.sbatch           # A100 (renyi)
#   sbatch $J -p gpu --gres=gpu:v100:1 cluster/v2/hello.sbatch           # V100
#
# It checks that the pinned clang compiles and runs OpenMP, MPI and (on GPU nodes) CUDA-driver code.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck disable=SC1091
source scripts/env.sh

echo "== node: $(hostname)  partition=${SLURM_JOB_PARTITION:-?}  gpus=${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-none}}"
echo "== clang:   $("$CXX" --version | head -1)"
echo "== llvm:    $(llvm-config --version) (assertions: $(llvm-config --assertion-mode))"
echo "== libomp:  $(ls "$OPSJIT_OMP_LIB_DIR"/libomp.so)"
echo "== mpicc:   $(command -v mpicc || echo missing)"
echo "== nvcc:    $(command -v nvcc || echo missing)"

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

# 1. OpenMP with the pinned clang and LLVM libomp (what the JIT's omp dialect lowering links against).
cat >"$T/omp.cpp" <<'EOF'
#include <omp.h>
#include <cstdio>
int main() {
  long s = 0;
#pragma omp parallel for reduction(+ : s)
  for (int i = 0; i < 1000; ++i) s += i;
  std::printf("omp threads=%d sum=%ld\n", omp_get_max_threads(), s);
  return s == 499500 ? 0 : 1;
}
EOF
"$CXX" -fopenmp -O2 "$T/omp.cpp" -o "$T/omp" -Wl,-rpath,"$OPSJIT_OMP_LIB_DIR"
"$T/omp"
ldd "$T/omp" | grep -E "libomp|libgomp" || true

# 2. MPI (host C++ compiled by the pinned clang against the module's OpenMPI).
if [ -n "${MPI_INSTALL_PATH:-}" ] && [ -x "$MPI_INSTALL_PATH/bin/mpirun" ]; then
  cat >"$T/mpi.cpp" <<'EOF'
#include <mpi.h>
#include <cstdio>
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int r, n; MPI_Comm_rank(MPI_COMM_WORLD, &r); MPI_Comm_size(MPI_COMM_WORLD, &n);
  int one = 1, sum = 0; MPI_Allreduce(&one, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (r == 0) std::printf("mpi ranks=%d allreduce=%d\n", n, sum);
  MPI_Finalize();
  return sum == n ? 0 : 1;
}
EOF
  "$CXX" -O2 "$T/mpi.cpp" -o "$T/mpi" -I"$MPI_INSTALL_PATH/include" -L"$MPI_INSTALL_PATH/lib" -lmpi \
    -Wl,-rpath,"$MPI_INSTALL_PATH/lib"
  mpirun --oversubscribe -np 2 "$T/mpi"
else
  echo "== MPI not available on this host; skipped"
fi

# 3. GPU: driver visible, CUDA driver API usable from code built by the pinned clang.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  cat >"$T/cu.cpp" <<'EOF'
#include <cuda.h>
#include <cstdio>
int main() {
  if (cuInit(0) != CUDA_SUCCESS) { std::puts("cuInit failed"); return 1; }
  CUdevice d; cuDeviceGet(&d, 0);
  int major = 0, minor = 0;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, d);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, d);
  char name[128]; cuDeviceGetName(name, sizeof name, d);
  std::printf("cuda driver ok: %s sm_%d%d\n", name, major, minor);
  return 0;
}
EOF
  "$CXX" -O2 "$T/cu.cpp" -o "$T/cu" -I"$CUDA_INSTALL_PATH/include" -L"$CUDA_INSTALL_PATH/lib64/stubs" -lcuda
  "$T/cu"
else
  echo "== no GPU on this node; CUDA check skipped"
fi
echo "HELLO OK"
