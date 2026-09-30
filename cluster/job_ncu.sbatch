#!/bin/bash
#SBATCH --job-name=opsmlir-ncu
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.log
# Nsight Compute profile of one Taylor-Green Runge-Kutta stage (single precision, 128^3),
# unfused and with the default planner. Writes ncu_unfused.csv / ncu_fused.csv to
# $OPS_NCU_OUT (default: $OPS_CL_HOME/results). Summarise them with
# apps/c/taylor_green_vortex/ncu_summary.py.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env_cl.sh"
OUT=${OPS_NCU_OUT:-$OPS_CL_HOME/results}; mkdir -p "$OUT"
B=$OPS_CL_ROOT/ops-mlir/build/apps/c/taylor_green_vortex
WORK=$(mktemp -d "$OPS_CL_TMP/ncu.XXXX"); cd "$WORK"
M=launch__registers_per_thread,launch__occupancy_limit_registers,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,gpu__time_duration.sum,lts__t_sector_hit_rate.pct,launch__grid_size,launch__block_size
# launch 0 is the initial-condition kernel; the next 17 (unfused) / 6 (fused) are stage 1.
for cfg in "unfused 0 17" "fused 1 6"; do set -- $cfg
  OPS_MLIR_FUSION=$2 OPS_BACKEND=cuda TGV_N=${TGV_N:-128} TGV_NITER=2 \
    ncu --csv --metrics $M --launch-skip 1 --launch-count $3 "$B/opensbli_f32" \
    > "$OUT/ncu_$1.csv" 2> "$WORK/ncu_$1.err"
done
