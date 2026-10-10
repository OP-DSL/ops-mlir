#!/bin/bash -l
#SBATCH --job-name=opsjit-ctest
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log
# Run the test suites. The first argument is a ctest label expression:
#   sbatch -D $PWD -o $HOME/ops-jit-cl/logs/%x-%j.log cluster/v2/ctest.sbatch cpu
#   sbatch -D $PWD -o $HOME/ops-jit-cl/logs/%x-%j.log -p gpu --gres=gpu:a100:1 cluster/v2/ctest.sbatch gpu
#   sbatch ... cluster/v2/ctest.sbatch mpi          # runs mpirun -np 2/4 tests
# Labels: cpu, gpu, mpi (see docs/design.md, test strategy). Also runs `ninja check-ops-mlir` for "cpu".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck disable=SC1091
source scripts/env.sh
label=${1:-cpu}
BUILD=${OPSJIT_BUILD_DIR:-$PWD/build}
if [ ! -f "$BUILD/CTestTestfile.cmake" ]; then echo "ctest.sbatch: nothing built in $BUILD yet" >&2; exit 2; fi
if [ "$label" = cpu ]; then ninja -C "$BUILD" check-ops-mlir; fi
ctest --test-dir "$BUILD" -L "$label" --output-on-failure -j "${SLURM_CPUS_PER_TASK:-4}"
