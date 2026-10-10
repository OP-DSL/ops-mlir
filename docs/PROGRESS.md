# Implementation progress log

Running log of `IMPLEMENTATION_PLAN.md` execution, kept for the final review. One entry per task:
what was done, deviations from the plan, and facts discovered that the plan did not know.
Newest entries at the bottom. Cluster = `cl` (SLURM); local = WSL Ubuntu 24.04.

## Environment facts (2026-10-10)

- Local `llvm-project/build` (23.1.2, main snapshot) is **unusable**: it was copied through Windows, so all
  executables lost their exec bit and the 32 symlinks in `bin/` and `lib/` (clang, clang++, ...) became
  0-byte files. It is not used by v2. (`chmod +x` fixed `mlir-opt`, but not the symlinks.)
- The `ops-mlir-work` source trees (OPS, ops-mlir, xdsl, llvm-project) carry `*:Zone.Identifier` files and
  broken refs `refs/heads/*:Zone.Identifier` from the same Windows copy. They are ignored through
  `.git/info/exclude`; deleting them is pending Istvan's OK (plan P0.1).
- The local OPS clone was shallow (1 commit); it was un-shallowed from `origin` so the final review has a history.
- The Bash tool of the harness is Git Bash on Windows; WSL commands go through `wsl.exe` (PowerShell).
  Scripts piped to `bash -s` get a trailing `\r` on the last line from PowerShell: end them with a sentinel.
- Local machine: 16 cores, 15 GB RAM, RTX 4050 (sm_89, driver visible in WSL), `nvcc` 12.0, no MPI, no
  `gfortran`, no `sudo`. MPI/HDF5 tests therefore run on the cluster.

## P0.0 Toolchain pin and cluster bootstrap

- **Pin:** LLVM/MLIR/clang **23.1.0** = the existing cluster install
  `~/MLIR/lib_install/llvm-project-23.1.0-CUDA-Release` (assertions on, `X86;NVPTX`, includes lld, the LLVM
  OpenMP runtime `lib/x86_64-unknown-linux-gnu/libomp.so`, `libmlir_cuda_runtime.so`, FileCheck/not/count,
  mlir-runner; no `llvm-lit`). It was rsync-copied (5.4 GB) to `~/ops-mlir-work/toolchain/llvm-23.1.0`, so
  local and cluster use byte-identical toolchains. `cmake/LLVMVersion.cmake` fails the configure on a
  different release series.
- `scripts/env.sh` detects local vs cluster and exports the same variables on both (`LLVM_DIR`, `MLIR_DIR`,
  `CC/CXX`, `OPS_SRC`, `OPS_INSTALL`, CUDA/MPI/HDF5 paths). On the cluster it enables gcc-toolset-13 (libstdc++ the
  clang binaries need) and loads `openmpi/4.1.6` and `OP2-env` (CUDA 12.3, HDF5, MPI paths).
- Cluster: `~/ops-jit-cl/{OPS.git,ops-mlir.git,work,logs,results}`; local remotes `cl-ops`, `cl-mlir`;
  `cluster/v2/sync.sh` pushes the feature branches (never forced) and fast-forwards the work trees.
  SSH key copied into WSL (`~/.ssh/id_rsa_cl` + `Host cl` entry, with Windows CRLF line endings converted).
- **Acceptance:** `cluster/v2/hello.sbatch` (pinned clang + OpenMP/libomp + MPI + CUDA-driver check)
  passes on a CPU node (job 223629: `omp threads=4`, `mpi ranks=2 allreduce=2`, `HELLO OK`).
  A100 (223630) and V100 (223631) jobs are queued behind other users; results below when they run.
- **Deviation from the plan:** the plan's acceptance said "builds the OPS libraries with the pinned clang"; the
  hello job checks the toolchain only, and the OPS build is P0.2 (`scripts/build_ops.sh`).
- Notes: `.gitignore` ignored `*.cmake`; `!cmake/*.cmake` added. `core.fileMode=false` in OPS and ops-mlir
  (Windows-copy mode noise), so exec bits are recorded with `git update-index --chmod=+x`.
