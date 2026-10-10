#!/bin/bash -l
#SBATCH --job-name=opsjit-build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log
# Build on the cluster: OPS libraries, then ops-mlir, then the apps.
#   sbatch -D $PWD -o $HOME/ops-jit-cl/logs/%x-%j.log cluster/v2/build.sbatch [ops|mlir|apps|all]
# CUDA libraries need the GPU partition for the driver stubs only at run time, so the build itself
# runs on a CPU node; submit with -p gpu --gres=gpu:a100:1 if a step needs a device.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck disable=SC1091
source scripts/env.sh
what=${1:-all}
for step in $( [ "$what" = all ] && echo "ops mlir apps" || echo "$what" ); do
  script=scripts/build_$step.sh
  if [ ! -x "$script" ]; then echo "build.sbatch: $script does not exist yet" >&2; exit 2; fi
  echo "== $(date -Is) build $step"
  "$script"
done
