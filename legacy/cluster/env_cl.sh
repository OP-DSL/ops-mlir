#!/usr/bin/env bash
# Environment for building and running ops-mlir on the `cl` cluster's renyi node
# (2x NVIDIA A100-40GB, RHEL 8, CUDA 12.3). Source it ON renyi (inside a Slurm job):
#
#     source cluster/env_cl.sh
#
# Everything lives under the shared home directory, so the login node and any compute
# node see the same tree (nothing is kept on renyi's node-local /scratch).
# Override any of these before sourcing.

# Staging area: the git remote renyi pulls from, job logs, results.
export OPS_CL_HOME=${OPS_CL_HOME:-$HOME/ops-mlir-cl}
# Work tree (sources, venv, build).
export OPS_CL_ROOT=${OPS_CL_ROOT:-$OPS_CL_HOME/work}
# Transient per-run directories (the evaluation creates and deletes hundreds). Each run
# writes an HDF5 output file, up to ~1 GB at the largest grids; point this at a node-local
# disk if NFS write speed or the nearly full /home becomes a problem.
export OPS_CL_TMP=${OPS_CL_TMP:-$OPS_CL_ROOT/tmp}
mkdir -p "$OPS_CL_TMP" 2>/dev/null || true

# Host toolchain: gcc-toolset-13 provides the libstdc++ that the clang 23 binaries
# below were built against and that ops-mlir must link with.
# shellcheck disable=SC1091
source /opt/rh/gcc-toolset-13/enable

# CUDA 12.3 toolkit (ptxas, libdevice, libcudart). The driver (580.x) is newer.
export CUDA_INSTALL_PATH=${CUDA_INSTALL_PATH:-/home/shared/software/cuda/12.3}
export CUDA_ROOT=$CUDA_INSTALL_PATH CUDA_HOME=$CUDA_INSTALL_PATH

# An existing LLVM/MLIR/clang 23.1.0 *install* (mlir, clang, lld, openmp, CUDA runner).
# Used read-only. For an installed LLVM the "build", "source" and install dirs coincide.
export LLVM_INSTALL_PATH=${LLVM_INSTALL_PATH:-$HOME/MLIR/lib_install/llvm-project-23.1.0-CUDA-Release}
export MLIR_BUILD_DIR=$LLVM_INSTALL_PATH
export LLVM_BUILD_DIR=$LLVM_INSTALL_PATH
export LLVM_SOURCE_DIR=$LLVM_INSTALL_PATH
export MLIR_SOURCE_DIR=$LLVM_INSTALL_PATH
export OPS_OMP_INCLUDE_DIR=$LLVM_INSTALL_PATH/lib/clang/23/include
export OPS_OMP_LIB_DIR=$LLVM_INSTALL_PATH/lib/x86_64-unknown-linux-gnu
export OPS_MLIR_CUDA_RUNTIME=$LLVM_INSTALL_PATH/lib/libmlir_cuda_runtime.so

# OPS (the C library) and serial HDF5 (the TGV app writes HDF5).
export OPS_ROOT=$OPS_CL_ROOT/OPS/ops/c
export HDF5_INSTALL_PATH=${HDF5_INSTALL_PATH:-/home/shared/software/op-dsl/hdf5-seq}

export PATH=$LLVM_INSTALL_PATH/bin:$CUDA_INSTALL_PATH/bin:$PATH
# /lib64 FIRST, on purpose. Linking libpython3.12 from conda puts conda's lib directory
# in the executable's RUNPATH, so without this the executable loads conda's libgcc_s and
# libstdc++ instead of the system ones the LLVM libraries were built against. Two
# libgcc_s copies then disagree about the JIT's registered exception frames and the
# process aborts in __deregister_frame at exit -- silently and only now and then
# (about 1 run in 8 on renyi; 0 in 60 with this line).
export LD_LIBRARY_PATH=/lib64:/opt/rh/gcc-toolset-13/root/usr/lib64:$CUDA_INSTALL_PATH/lib64:$LLVM_INSTALL_PATH/lib:$OPS_OMP_LIB_DIR:$HDF5_INSTALL_PATH/lib:${LD_LIBRARY_PATH:-}

# Python for the embedded xDSL lowering: conda's Python 3.12 (has libpython3.12.so),
# with the xDSL fork in a venv. The embedded interpreter finds the venv via PYTHONPATH.
export OPS_CL_PYTHON=${OPS_CL_PYTHON:-/home/reguly/miniconda3/bin/python3}
if [ -f "$OPS_CL_ROOT/venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$OPS_CL_ROOT/venv/bin/activate"
  export PYTHONPATH=$(python -c 'import site;print(site.getsitepackages()[0])'):$OPS_CL_ROOT/xdsl:${PYTHONPATH:-}
fi

# Sensible OpenMP defaults for the CPU backend.
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}
