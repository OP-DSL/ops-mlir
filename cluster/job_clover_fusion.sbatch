#!/bin/bash
#SBATCH --job-name=opsmlir-clover-fusion
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
# Fusion evaluation on CloverLeaf 2D and 3D: launches and steady-state time per variant.
#   sbatch -D ~/ops-mlir-cl/src -o ~/ops-mlir-cl/logs/%x-%j.log cluster/job_clover_fusion.sbatch
# Results: ~/ops-mlir-cl/work/eval/clover_fusion_{2d,3d}.json
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env_cl.sh"
flock "$OPS_CL_ROOT/.setup.lock" bash "$SLURM_SUBMIT_DIR/cluster/setup_renyi.sh"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
mkdir -p "$OPS_CL_ROOT/eval"
cd "$OPS_CL_ROOT/ops-mlir"
V=off,consecutive,dag,dag-latest,dag-max32
python3 apps/c/cloverleaf_2d/eval_fusion.py --exe build/apps/c/cloverleaf_2d/cloverleaf_2d \
  --deck "$OPS_CL_ROOT/OPS/apps/c/CloverLeaf/clover_bm_short.in" --backends seq,openmp,cuda \
  --variants $V --threads 16 --tmp "$OPS_CL_TMP" --out "$OPS_CL_ROOT/eval/clover_fusion_2d.json"
python3 apps/c/cloverleaf_2d/eval_fusion.py --exe build/apps/c/cloverleaf_3d/cloverleaf_3d \
  --deck "$OPS_CL_ROOT/OPS/apps/c/CloverLeaf_3D/clover_bm_short.in" --backends openmp,cuda \
  --variants $V --threads 16 --tmp "$OPS_CL_TMP" --out "$OPS_CL_ROOT/eval/clover_fusion_3d.json"
