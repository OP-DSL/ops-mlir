#!/usr/bin/env bash
# CloverLeaf runs for the evaluation: the stock implementation and ops-mlir on
# seq / OpenMP / CUDA, each checked with CloverLeaf's built-in QA.
#
#   cluster/run_cloverleaf.sh <2d|3d> <default|bm_short|bm> [backends...]
#
# Run it on renyi (inside a Slurm job, see job_cloverleaf.sbatch). One line per
# configuration goes to stdout:  <config> <QA %> <PASSED|FAILED> <wall s> <ops-mlir stats>
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env_cl.sh"

DIM=${1:-2d}; DECK=${2:-default}; shift 2 || true
BACKENDS=${*:-"stock seq openmp cuda"}
BUILD=$OPS_CL_ROOT/ops-mlir/build/apps/c/cloverleaf_$DIM
case $DECK in
  default) IN=clover.in_default ;;
  bm_short) IN=clover_bm_short.in ;;
  bm) IN=clover_bm.in ;;
  *) IN=clover_$DECK.in ;;
esac
SRC=$OPS_CL_ROOT/OPS/apps/c/$([ "$DIM" = 2d ] && echo CloverLeaf || echo CloverLeaf_3D)
THREADS=${OMP_NUM_THREADS:-16}

for be in $BACKENDS; do
  dir=$(mktemp -d "$OPS_CL_TMP/clover.XXXXXX")
  cp "$SRC/$IN" "$dir/clover.in"
  cd "$dir"
  if [ "$be" = stock ]; then exe=$BUILD/cloverleaf_${DIM}_stock; env=(OMP_NUM_THREADS=1)
  else exe=$BUILD/cloverleaf_$DIM; env=(OPS_BACKEND=$be OMP_NUM_THREADS=$THREADS OPS_MLIR_STATS=1); fi
  start=$(date +%s.%N)
  env "${env[@]}" "$exe" > out.txt 2> err.txt
  rc=$?
  end=$(date +%s.%N)
  qa=$(grep -o 'is within [0-9.E+-]*' out.txt | awk '{print $3}')
  verdict=$(grep -o 'PASSED\|FAILED' out.txt | tail -1)
  stats=$(grep -h 'ops-mlir coverage\|ops-mlir stats' err.txt | tr '\n' ' ' | cut -c1-400)
  printf '%s/%s/%s rc=%s qa=%s %s wall=%.1f %s\n' "$DIM" "$DECK" "$be" "$rc" "${qa:-none}" "${verdict:-none}" \
         "$(echo "$end - $start" | bc)" "$stats"
  cd - >/dev/null; rm -rf "$dir"
done
