#!/bin/bash
#SBATCH --job-name=opsmlir-clover-host
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.log
# Where the non-kernel time goes on the larger decks: which loops stay on the host, what they cost,
# and what the device-to-host copies ahead of them cost (default fusion, CUDA).
# Results: ~/ops-mlir-cl/work/eval/clover_host_{2d,3d}.json
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env_cl.sh"
flock "$OPS_CL_ROOT/.setup.lock" bash "$SLURM_SUBMIT_DIR/cluster/setup_renyi.sh"
E=$OPS_CL_ROOT/eval; mkdir -p "$E"
cd "$OPS_CL_ROOT/ops-mlir"
EF="python3 apps/c/cloverleaf_2d/eval_fusion.py --tmp $OPS_CL_TMP --backends cuda --variants dag"
$EF --exe build/apps/c/cloverleaf_2d/cloverleaf_2d --deck $OPS_CL_ROOT/OPS/apps/c/CloverLeaf/clover_bm16_short.in --out $E/clover_host_2d.json
$EF --exe build/apps/c/cloverleaf_3d/cloverleaf_3d --deck $OPS_CL_ROOT/OPS/apps/c/CloverLeaf_3D/clover_bm1s_short.in --out $E/clover_host_3d.json
