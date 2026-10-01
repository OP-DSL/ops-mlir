#!/bin/bash
#SBATCH --job-name=opsmlir-eval
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.log
# Run the Taylor-Green vortex evaluation on one A100 and write results.json.
#   sbatch cluster/job_eval.sbatch [extra args for eval_tgv.py run]
set -euo pipefail
# Bring the checkout and build up to date first (cheap when nothing changed).
[ "${OPS_SKIP_SETUP:-0}" = 1 ] || bash "$SLURM_SUBMIT_DIR/cluster/setup_renyi.sh"
source "$SLURM_SUBMIT_DIR/cluster/env_cl.sh"
OUT=${OPS_EVAL_OUT:-$OPS_CL_ROOT/eval}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
cd "$OPS_CL_ROOT/ops-mlir"
python apps/c/taylor_green_vortex/eval_tgv.py run \
  --app build/apps/c/taylor_green_vortex --out "$OUT/tgv_eval.json" --work "$OUT/work" "$@"
