#!/bin/bash
#SBATCH --job-name=opsmlir-clover-large
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.log
# The larger CloverLeaf decks (2D 3840^2: clover_bm16_short.in, 3D 256^3: clover_bm1s_short.in):
# stock baseline (1 thread, in the background), CUDA fusion-size sweeps with launch logs, then
# OpenMP. Results: ~/ops-mlir-cl/work/eval/clover_large_*.json, launch logs in eval/launch_logs.
#   sbatch -D ~/ops-mlir-cl/src -o ~/ops-mlir-cl/logs/%x-%j.log cluster/job_clover_large.sbatch
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env_cl.sh"
bash "$SLURM_SUBMIT_DIR/cluster/setup_renyi.sh"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
E=$OPS_CL_ROOT/eval; mkdir -p "$E"
cd "$OPS_CL_ROOT/ops-mlir"
B=build/apps/c
D2=$OPS_CL_ROOT/OPS/apps/c/CloverLeaf/clover_bm16_short.in
D3=$OPS_CL_ROOT/OPS/apps/c/CloverLeaf_3D/clover_bm1s_short.in
EF="python3 apps/c/cloverleaf_2d/eval_fusion.py --tmp $OPS_CL_TMP"

# stock baselines, one thread each, on a CPU core of their own while the GPU runs go
( $EF --exe $B/cloverleaf_2d/cloverleaf_2d --stock-exe $B/cloverleaf_2d/cloverleaf_2d_stock --deck $D2 --backends "" --out $E/clover_large_stock_2d.json
  $EF --exe $B/cloverleaf_3d/cloverleaf_3d --stock-exe $B/cloverleaf_3d/cloverleaf_3d_stock --deck $D3 --backends "" --out $E/clover_large_stock_3d.json ) &
STOCK=$!

CUDA_V=off,consecutive,max2,max3,max4,max8,max16,max64,ratio0.5,ratio2
$EF --exe $B/cloverleaf_2d/cloverleaf_2d --deck $D2 --backends cuda --variants $CUDA_V \
    --launch-logs $E/launch_logs_2d --out $E/clover_large_cuda_2d.json
$EF --exe $B/cloverleaf_3d/cloverleaf_3d --deck $D3 --backends cuda --variants $CUDA_V \
    --launch-logs $E/launch_logs_3d --out $E/clover_large_cuda_3d.json

wait $STOCK
OMP_V=off,consecutive,max4,max8,max64
$EF --exe $B/cloverleaf_2d/cloverleaf_2d --deck $D2 --backends openmp --variants $OMP_V --threads 16 --out $E/clover_large_openmp_2d.json
$EF --exe $B/cloverleaf_3d/cloverleaf_3d --deck $D3 --backends openmp --variants $OMP_V --threads 16 --out $E/clover_large_openmp_3d.json
