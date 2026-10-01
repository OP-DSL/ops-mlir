#!/usr/bin/env bash
# Set up ops-mlir on renyi: clone the sources, create the xDSL venv, configure and
# build. Idempotent: rerun it to pick up new commits and rebuild.
#
# Run it inside a Slurm job on renyi (see cluster/job_build.sbatch), because the
# work tree is on renyi's node-local /scratch.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$HERE/env_cl.sh"

# Pinned to the versions the local development used.
OPS_URL=${OPS_URL:-https://github.com/OP-DSL/OPS.git};      OPS_REF=${OPS_REF:-c0af0f124469e5fd856b594a23ff1206c3e9c7a8}
XDSL_URL=${XDSL_URL:-https://github.com/Archii0/xdsl.git};  XDSL_REF=${XDSL_REF:-053b120e0b29a2fe0c421be29143bad98ad4e77f}
OPSMLIR_REMOTE=${OPSMLIR_REMOTE:-$OPS_CL_HOME/ops-mlir.git}
OPSMLIR_BRANCH=${OPSMLIR_BRANCH:-main}
JOBS=${JOBS:-$(nproc)}

mkdir -p "$OPS_CL_ROOT"
cd "$OPS_CL_ROOT"

clone_at() {   # <url> <dir> <ref>
  if [ ! -d "$2/.git" ]; then git clone --quiet "$1" "$2"; fi
  git -C "$2" fetch --quiet origin
  git -C "$2" checkout --quiet "$3"
}

echo "== sources"
clone_at "$OPS_URL" OPS "$OPS_REF"
clone_at "$XDSL_URL" xdsl "$XDSL_REF"
if [ ! -d ops-mlir/.git ]; then git clone --quiet "$OPSMLIR_REMOTE" ops-mlir; fi
git -C ops-mlir fetch --quiet origin
git -C ops-mlir checkout --quiet "$OPSMLIR_BRANCH"
git -C ops-mlir reset --quiet --hard "origin/$OPSMLIR_BRANCH"
echo "ops-mlir at $(git -C ops-mlir log -1 --format='%h %s')"

echo "== python venv (xDSL fork)"
if [ ! -f venv/bin/activate ]; then "$OPS_CL_PYTHON" -m venv venv; fi
# shellcheck disable=SC1091
source "$OPS_CL_ROOT/venv/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e xdsl
# Re-source so PYTHONPATH picks up the venv.
source "$HERE/env_cl.sh"
python -c "import xdsl, sys; print('xdsl', xdsl.__file__); print(sys.version.split()[0])"

echo "== configure ops-mlir"
cmake -G Ninja -S ops-mlir -B ops-mlir/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$LLVM_INSTALL_PATH/bin/clang++" \
  -DCMAKE_C_COMPILER="$LLVM_INSTALL_PATH/bin/clang" \
  -DOPS_ENABLE_CUDA=ON \
  -DCUDAToolkit_ROOT="$CUDA_INSTALL_PATH" \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DPython3_ROOT_DIR="$(dirname "$(dirname "$OPS_CL_PYTHON")")"

echo "== build ($JOBS jobs)"
cmake --build ops-mlir/build -j"$JOBS"
echo "== done"
