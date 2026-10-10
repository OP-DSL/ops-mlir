#!/usr/bin/env bash
# Publish the local feature branches to the `cl` cluster and update the cluster work trees.
# Run LOCALLY (WSL), from anywhere:
#
#     cluster/v2/sync.sh
#
# Pushes go to the staging bare repos on the cluster (~/ops-jit-cl/{OPS,ops-mlir}.git), never to
# GitHub, and are never forced: if a push is rejected, make a new commit instead of amending.
set -euo pipefail

WORK=${OPSJIT_LOCAL_WORK:-$HOME/ops-mlir-work}
OPS_BRANCH=${OPS_BRANCH:-feature/jit-executor}
MLIR_BRANCH=${MLIR_BRANCH:-v2}

git -C "$WORK/OPS" push cl-ops "$OPS_BRANCH:$OPS_BRANCH"
git -C "$WORK/ops-mlir" push cl-mlir "$MLIR_BRANCH:$MLIR_BRANCH"

# The remote script comes from this heredoc, so no `ssh -n` here (it would replace stdin).
ssh cl bash -s -- "$OPS_BRANCH" "$MLIR_BRANCH" <<'REMOTE'
set -euo pipefail
OPS_BRANCH=$1; MLIR_BRANCH=$2
W=$HOME/ops-jit-cl/work
mkdir -p "$W"
update() {  # update <name> <branch>
  local name=$1 branch=$2
  if [ ! -d "$W/$name/.git" ]; then
    git clone -q "$HOME/ops-jit-cl/$name.git" "$W/$name"
  fi
  git -C "$W/$name" fetch -q origin
  git -C "$W/$name" checkout -q "$branch" 2>/dev/null || git -C "$W/$name" checkout -q -b "$branch" "origin/$branch"
  git -C "$W/$name" merge -q --ff-only "origin/$branch"
  echo "$name @ $(git -C "$W/$name" log -1 --format='%h %s')"
}
update OPS "$OPS_BRANCH"
update ops-mlir "$MLIR_BRANCH"
REMOTE
