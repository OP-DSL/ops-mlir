# ops-mlir on the `cl` cluster (renyi, 2x A100)

Everything runs on **renyi** (Slurm partition `gpu`, `--gres=gpu:a100:N`). The work tree
is on renyi's node-local `/scratch` (fast; the NFS home is nearly full), so every step
that touches it is a Slurm job.

| What | Where |
|---|---|
| staging git remote (push here) | `~/ops-mlir-cl/ops-mlir.git` on the login node |
| job scripts + logs | `~/ops-mlir-cl/src/cluster/`, `~/ops-mlir-cl/logs/` |
| sources, venv, build (renyi only) | `/scratch/$USER/ops-mlir-work/{OPS,xdsl,venv,ops-mlir}` |
| LLVM / MLIR / clang 23.1.0 (read-only, not rebuilt) | `~/MLIR/lib_install/llvm-project-23.1.0-CUDA-Release` |
| CUDA 12.3, serial HDF5, Python 3.12 | `/home/shared/software/cuda/12.3`, `.../op-dsl/hdf5-seq`, conda |

## Workflow

```bash
# from the dev machine: publish the current commit
git push cl:ops-mlir-cl/ops-mlir.git HEAD:refs/heads/main

# on the login node (ssh cl): refresh the job scripts, then submit
cd ~/ops-mlir-cl/src && git pull
J='-D ~/ops-mlir-cl/src -o ~/ops-mlir-cl/logs/%x-%j.log'
sbatch $J cluster/job_build.sbatch                    # clone, venv, configure, build
sbatch $J cluster/job_tests.sbatch                    # full ctest on an A100
sbatch $J cluster/job_eval.sbatch --main-n 256 --cpu-n 64 --omp-threads 24 \
       --peak-gbs 1555 --sizes 32,64,96,128,192,256,320,384 --sizes-f64 64,128,192,256
```

`job_eval.sbatch` runs `setup_renyi.sh` first, so a push followed by an eval submission
rebuilds incrementally. Results land in `/scratch/$USER/ops-mlir-work/eval/tgv_eval.json`;
copy the file back and run `eval_tgv.py report --results ... --outdir docs --tag a100`.

## Things that bit us

* **`libgcc_s` / `libstdc++` from conda.** `libpython3.12` comes from conda, which puts
  conda's `lib/` in the executable's RUNPATH. The executable then loads conda's `libgcc_s`
  and `libstdc++` instead of the system ones, and the JIT's exception-frame
  registration breaks: a silent `abort()` in `__deregister_frame` at exit, about 1 run
  in 8 (it hides under `gdb`; `OPS_MLIR_CRASH_TRACE=1` prints the backtrace).
  `env_cl.sh` puts `/lib64` first in `LD_LIBRARY_PATH`; keep that when writing new
  job scripts.
* **Shared LLVM install.** It is only read. It has no NVPTX-compiler/fatbin libraries, so
  GPU code is assembled with `ptxas` from CUDA 12.3 (`CUDA_ROOT` is set in `env_cl.sh`).
* **GPU architecture** is detected from the device (`sm_80` on the A100s).

## Results

The A100 evaluation (`docs/tgv_evaluation_a100.md`, data in `docs/data/tgv_eval_a100.json`) was
produced by `job_eval.sbatch` with the arguments shown above, plus `job_ncu.sbatch` for the
Nsight Compute section. The results file lives on renyi's `/scratch`; copy it to the shared home
from inside a job (`srun -p gpu -w renyi cp ...`) before fetching it.
