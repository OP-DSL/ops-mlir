#!/usr/bin/env bash
# Environment for building and running ops-mlir on the `cl` cluster's renyi node
# (2x NVIDIA A100-40GB, RHEL 8, CUDA 12.3). Source it ON renyi (inside a Slurm job):
#
#     source cluster/env_cl.sh
#
# Everything lives on renyi's node-local /scratch (fast, unlike the NFS home).
# Override any of these before sourcing.

# Work tree (sources, venv, build) on node-local disk.
export OPS_CL_ROOT=${OPS_CL_ROOT:-/scratch/$USER/ops-mlir-work}
# Staging area on the shared (NFS) home: the git remote renyi pulls from, and job logs.
export OPS_CL_HOME=${OPS_CL_HOME:-$HOME/ops-mlir-cl}

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
# Order matters: gcc-toolset's libstdc++ must win over conda's older one.
export LD_LIBRARY_PATH=/opt/rh/gcc-toolset-13/root/usr/lib64:$CUDA_INSTALL_PATH/lib64:$LLVM_INSTALL_PATH/lib:$OPS_OMP_LIB_DIR:$HDF5_INSTALL_PATH/lib:${LD_LIBRARY_PATH:-}

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
