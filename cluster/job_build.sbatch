#!/bin/bash
#SBATCH --job-name=opsmlir-build
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log
# Clone, configure and build ops-mlir on renyi. Submit from the repo root (or with
# --chdir) as:  sbatch cluster/job_build.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
bash cluster/setup_renyi.sh
