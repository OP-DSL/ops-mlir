#!/bin/bash -l
#SBATCH --job-name=opsjit-run
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log
# Run an arbitrary command inside the ops-mlir environment on a compute node. The thin
# build.sbatch / ctest.sbatch / perf.sbatch wrappers call this; use it directly for one-offs:
#
#   cd ~/ops-jit-cl/work/ops-mlir
#   sbatch -D $PWD -o $HOME/ops-jit-cl/logs/%x-%j.log -p gpu --gres=gpu:a100:1 \
#          cluster/v2/run.sbatch ./scripts/some_script.sh arg1 arg2
#
# Etiquette: modest --time, *_lowpriority partitions for long sweeps, never hold both A100s for
# more than an hour at a time, never cancel other users' jobs.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck disable=SC1091
source scripts/env.sh
echo "== $(date -Is) on $(hostname): $*"
"$@"
echo "== $(date -Is) done"
