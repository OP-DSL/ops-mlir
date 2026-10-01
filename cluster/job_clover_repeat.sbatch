#!/bin/bash
#SBATCH --job-name=opsmlir-clover-repeat
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.log
# CUDA fusion variants on the larger CloverLeaf decks, repeated and interleaved, with nothing else of
# ours running (the first sweep overlapped the long single-thread stock baseline and renyi is shared
# with other users). Results: ~/ops-mlir-cl/work/eval/clover_repeat_{2d,3d}.json
#   sbatch -D $HOME/ops-mlir-cl/src -o $HOME/ops-mlir-cl/logs/%x-%j.log cluster/job_clover_repeat.sbatch
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env_cl.sh"
flock "$OPS_CL_ROOT/.setup.lock" bash "$SLURM_SUBMIT_DIR/cluster/setup_renyi.sh"
E=$OPS_CL_ROOT/eval; mkdir -p "$E"
cd "$OPS_CL_ROOT/ops-mlir"
V=off,consecutive,max2,max3,max4,noguard,ratio16
EF="python3 apps/c/cloverleaf_2d/eval_fusion.py --tmp $OPS_CL_TMP --backends cuda --variants $V --repeat ${REPEAT:-3}"
$EF --exe build/apps/c/cloverleaf_2d/cloverleaf_2d --deck $OPS_CL_ROOT/OPS/apps/c/CloverLeaf/clover_bm16_short.in --launch-logs $E/repeat_logs_2d --out $E/clover_repeat_2d.json
$EF --exe build/apps/c/cloverleaf_3d/cloverleaf_3d --deck $OPS_CL_ROOT/OPS/apps/c/CloverLeaf_3D/clover_bm1s_short.in --launch-logs $E/repeat_logs_3d --out $E/clover_repeat_3d.json
