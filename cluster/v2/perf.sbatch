#!/bin/bash -l
#SBATCH --job-name=opsjit-perf
#SBATCH --partition=cpu
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
# Performance sweeps: app x deck x variant, AOT versus JIT. Results go to ~/ops-jit-cl/results as JSON.
#   sbatch -D $PWD -o $HOME/ops-jit-cl/logs/%x-%j.log cluster/v2/perf.sbatch <suite>
#   sbatch ... -p gpu --gres=gpu:a100:1 cluster/v2/perf.sbatch <suite>
# A suite is a script scripts/perf/<suite>.sh that writes $OPSJIT_RESULTS/<suite>-<job>.json.
# --exclusive: timing runs must not share a node. Long sweeps: use the *_lowpriority partitions.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck disable=SC1091
source scripts/env.sh
suite=${1:?usage: perf.sbatch <suite>}
export OPSJIT_RESULTS=${OPSJIT_RESULTS:-$OPSJIT_WORK/results}
mkdir -p "$OPSJIT_RESULTS"
script=scripts/perf/$suite.sh
if [ ! -f "$script" ]; then echo "perf.sbatch: $script does not exist yet" >&2; exit 2; fi
export OPSJIT_JOB=${SLURM_JOB_ID:-local}
bash "$script"
