#!/bin/bash
#SBATCH --job-name=opsmlir-clover
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log
# CloverLeaf 2D (and 3D when present) on an A100: stock vs ops-mlir, CloverLeaf's QA.
#   sbatch -D ~/ops-mlir-cl/src -o ~/ops-mlir-cl/logs/%x-%j.log cluster/job_cloverleaf.sbatch [DIMS...]
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env_cl.sh"
bash "$SLURM_SUBMIT_DIR/cluster/setup_renyi.sh"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
DIMS=${*:-2d}
for d in $DIMS; do
  for deck in default bm_short; do
    bash "$SLURM_SUBMIT_DIR/cluster/run_cloverleaf.sh" "$d" "$deck" stock seq openmp cuda
  done
done
