#!/usr/bin/env bash
# Regenerates the raw dumps that docs/compilation_flow.md quotes.
#
#   docs/data/compilation_flow/capture.sh [OUTDIR]        (default: ./compilation_flow_dumps)
#
# Needs a built tree (build/), the xDSL environment (source ../env_setup, otherwise the xDSL
# stage fails with "No module named 'xdsl'"), and for the CUDA dump a GPU and OPS_MLIR_CUDA_RUNTIME.
# Every file written is stdout or stderr of one run; the listings in the document are functions or
# passes cut out of them (the *_IR dumps are large: OPS_DEBUG_PASS_IR prints the module after every pass).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
OUT=$(mkdir -p "${1:-compilation_flow_dumps}" && cd "${1:-compilation_flow_dumps}" && pwd)
TGV=$ROOT/build/apps/c/taylor_green_vortex/opensbli
CLV=$ROOT/build/apps/c/cloverleaf_2d/cloverleaf_2d
cd "$OUT"

echo "== TGV, N=16: plan, launch log, stencil IR, lowered IR (sequential)"
TGV_N=16 TGV_NITER=1 OPS_BACKEND=seq OPS_MLIR_PLAN=1 OPS_MLIR_STATS=1 OPS_MLIR_LAUNCH_LOG=tgv_launch.tsv \
  OPS_MLIR_DUMP_STENCIL=1 OPS_MLIR_DUMP_LOWERED=1 "$TGV" > tgv_seq.out 2> tgv_seq_ir.txt

echo "== TGV, every backend pass (sequential, OpenMP) and the optimised LLVM IR"
for be in seq openmp; do
  TGV_N=16 TGV_NITER=1 OPS_BACKEND=$be OPS_DEBUG_PASS_IR=1 OPS_MLIR_DUMP_LLVM=1 "$TGV" > tgv_$be.out 2> tgv_${be}_passes.txt
done

if [ -n "${OPS_MLIR_CUDA_RUNTIME:-}" ] && command -v nvidia-smi >/dev/null; then
  echo "== TGV, every backend pass (CUDA)"
  TGV_N=16 TGV_NITER=1 OPS_BACKEND=cuda OPS_DEBUG_PASS_IR=1 "$TGV" > tgv_cuda.out 2> tgv_cuda_passes.txt
else
  echo "== (skipping the CUDA dump: set OPS_MLIR_CUDA_RUNTIME and run on a GPU machine)"
fi

# CloverLeaf runs in a directory with a clover.in. The default deck is tiny (10x2 cells).
mkdir -p clover && cd clover
cp "$ROOT/../OPS/apps/c/CloverLeaf/clover.in_default" clover.in 2>/dev/null || \
  printf '*clover\n state 1 density=0.2 energy=1.0\n state 2 density=1.0 energy=2.5 geometry=rectangle xmin=0.0 xmax=5.0 ymin=0.0 ymax=2.0\n x_cells=10\n y_cells=2\n xmin=0.0\n ymin=0.0\n xmax=10.0\n ymax=2.0\n initial_timestep=0.04\n timestep_rise=1.5\n max_timestep=0.04\n end_time=3.0\n test_problem 1\n*endclover\n' > clover.in
run() {  # name, JIT_ONLY list, extra env...
  local name=$1 only=$2; shift 2
  env OPS_BACKEND=seq OPS_MLIR_JIT_ONLY="$only" OPS_MLIR_EXPLAIN=1 OPS_MLIR_PLAN=1 OPS_MLIR_DUMP_STENCIL=1 \
      OPS_MLIR_DUMP_LOWERED=1 "$@" "$CLV" > "$name.out" 2> "$name.txt"
  echo "   $name: $(grep -c '^===' "$name.txt") IR sections"
}
echo "== CloverLeaf 2D, one kernel (or pair) at a time; the others run on the host and act as barriers"
run guarded_pair   revert_kernel,accelerate_kernel
run reduction      calc_dt_kernel_min
run strided_write  initialise_chunk_kernel_x,initialise_chunk_kernel_xx
run struct_const   generate_chunk_kernel
run ideal_gas      ideal_gas_kernel OPS_DEBUG_PASS_IR=1     # the first pass dump contains the translated kernel body
run advec_selects  advec_cell_kernel3_xdir OPS_DEBUG_PASS_IR=1
echo "== CloverLeaf 2D, everything: plan of the initialisation queue and of the hydro step, launch log"
env OPS_BACKEND=seq OPS_MLIR_PLAN=1 OPS_MLIR_LAUNCH_LOG=launch.tsv OPS_MLIR_STATS=1 "$CLV" > full.out 2> full_plan.txt
echo "done: $OUT"
