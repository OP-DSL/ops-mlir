#!/bin/bash
#SBATCH --job-name=opsmlir-tests
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
# Run the whole ctest suite on an A100 (CPU backends and CUDA, fp32 and fp64).
set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env_cl.sh"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
cd "$OPS_CL_ROOT/ops-mlir"
ctest --test-dir build --output-on-failure "$@"
