#!/bin/bash
#SBATCH --job-name=opsmlir-clover-ncu
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.log
# Nsight Compute profile of one steady-state time step of the larger CloverLeaf decks, per
# fusion-size setting, joined to the loops each kernel fuses (apps/c/cloverleaf_2d/ncu_clover.py).
# Results: ~/ops-mlir-cl/work/eval/clover_ncu_{2d,3d}.json
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env_cl.sh"
bash "$SLURM_SUBMIT_DIR/cluster/setup_renyi.sh"
E=$OPS_CL_ROOT/eval; mkdir -p "$E"
cd "$OPS_CL_ROOT/ops-mlir"
V=${CLOVER_NCU_VARIANTS:-off,max2,max4,max8,max16,max64}
python3 apps/c/cloverleaf_2d/ncu_clover.py --exe build/apps/c/cloverleaf_2d/cloverleaf_2d \
  --deck $OPS_CL_ROOT/OPS/apps/c/CloverLeaf/clover_bm16_short.in --variants $V --tmp $OPS_CL_TMP --out $E/clover_ncu_2d.json
python3 apps/c/cloverleaf_2d/ncu_clover.py --exe build/apps/c/cloverleaf_3d/cloverleaf_3d \
  --deck $OPS_CL_ROOT/OPS/apps/c/CloverLeaf_3D/clover_bm1s_short.in --variants $V --tmp $OPS_CL_TMP --out $E/clover_ncu_3d.json
