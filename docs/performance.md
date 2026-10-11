# Performance

Measurements of the JIT (branch `v2`) against OPS's own ahead-of-time (AOT) code, on the `cl` cluster. The AOT executables are the standard
OPS builds of the same applications (`OPS-apps-new`: OpenMP, CUDA); the JIT executables are the same sources translated with `--jit`
(`apps/`, `scripts/build_apps.sh`). Both are built by the same clang 23. Times are the wall time of the whole run (start-up, the JIT's
compilation from the disk cache, output included), the best of 3 unless said otherwise. The JIT runs once before the timed runs to fill its
disk cache (`OPS_JIT_CACHE_DIR`); a cold start costs more, see "Compilation cost". Scripts: `scripts/perf/{cpu,gpu,cpu-kernels,cpu-groups,
cpu-fusion-sweep,gpu-blocks,sample}.sh` (through `cluster/v2/perf.sbatch`), raw results in `~/ops-jit-cl/results/`.

Machines: CPU nodes of the `cpu` partition (euclid: 48 cores; hajos, kemeny: 64 cores; one run per node, `--exclusive`, `OMP_PROC_BIND=close`,
`OMP_PLACES=cores`, as many threads as cores), GPU: NVIDIA A100 (40 GB, `renyi`), CUDA 12.3.

## CPU

| Case | Node | AOT [s] | JIT [s] | JIT, no fusion [s] | JIT / AOT |
|---|---|---:|---:|---:|---:|
| CloverLeaf 2D, 3840^2, 87 steps | hajos/kemeny (64 thr) | 27.4 | 27.7 | 28.9 | 1.01 |
| CloverLeaf 3D, 256^3, 87 steps | hajos/kemeny (64 thr) | 56.4 | **54.0** | 59.4 | 0.96 |
| TGV double, 512^3 | euclid (48 thr) | 1257 | **253** | 231 | 0.20 |
| TGV single | euclid (48 thr) | 1184 | **163** | 122 | 0.14 (0.10 unfused) |
| TGV single/double | hajos (64 thr) | 888 | **214** | 224 | 0.24 |

* CloverLeaf is at parity: per kernel (OPS's own timing table, `scripts/perf/cpu-kernels.sh`, 20 steps of the 3D case, 64 threads) the 38 heaviest
  kernels are within 2% of the AOT code; the loops that were slow before the fixes below were the loops of the 1-D coordinate dats.
* TGV (OpenSBLI, many large point-wise and stencil kernels per time step) is 4-10 times faster than the stock OpenMP code: the JIT fuses
  point-local chains and specialises the ranges and the constants; the stock code streams every temporary through memory.
* Fusion is not always a win on the CPU: TGV single is 34% faster *without* it (122 s against 163 s) and CloverLeaf 3D 10% slower. The
  fused groups of the DAG planner are large; there is no register or cache model yet (plan item P7: register-aware fusion). `OPS_JIT_FUSION_MAX`
  limits the group size; the sweep is in the section "Fusion size".

### What the first cluster run found

The first run (before the fixes listed in `PROGRESS.md`, "Performance investigation") had CloverLeaf 2D at 46.5 s against 18.1 s for AOT (euclid, 48
threads) and 3D at 124.8 s against 39.2 s. Causes, in the order of their weight:

1. OPS ran every loop that reads a 1-D (edge) dat after a write to it eagerly with the stock code and flushed the queue first; the rule is meant for
   the halo exchange of MPI runs, but its dirty bit is only cleared by the MPI code, so on one process it fired for ever. The executor never saw these
   loops (6% of them, the heavy advection and viscosity ones). OPS now applies the rule only with more than one process, and the JIT claims edge dats
   on one process.
2. A group that failed to lower ran with OPS: `calc_dt_kernel` of 3D (a conditional expression loading from two dats, whose branches the LLVM-dialect
   canonicaliser merged into a block with memref arguments): 2.7 s against 0.4 s per 20 steps. Pass order fixed.
3. The fusion planner's reachability test (a graph search per loop and candidate group) took seconds on the 500-loop chains of CloverLeaf 3D.
4. Constants and globals read inside an `if` of a kernel were not materialised (the group then ran with OPS).

## GPU (A100)

| Case | AOT CUDA [s] | JIT [s] | JIT, no fusion [s] | JIT / AOT |
|---|---:|---:|---:|---:|
| CloverLeaf 2D, 3840^2 | 5.88 | 6.46 | 6.57 | 1.10 |
| CloverLeaf 3D, 256^3 | 7.64 | 8.59 | 8.85 | 1.12 |
| TGV single, 512^3 | 13.16 | 14.47 | 19.50 | 1.10 (0.74 unfused) |

* TGV double and single/double do not fit in 40 GB with either code (AOT and JIT both run out of memory at the benchmark size).
* The thread block (`OPS_JIT_GPU_BLOCK`, default `32,4,1`, the one OPS uses) is the best or tied in a sweep of twelve shapes of 128 to 512 threads
  (`scripts/perf/gpu-blocks.sh`): 2D 6.45 s for every shape, 3D from 8.58 s (default) to 9.19 s, TGV from 14.47 s (default) to 16.17 s.
* The 10% are host side start-up: for CloverLeaf 3D the executor spends 2.96 s executing (kernel launches and reduction copies) of a 5.6 to 8.5 s
  run, and about 1 s on the plans (`frontend 0.25 s`, `plan passes 0.20 s`), the cache (`cache io 0.19 s`, loading 90 modules) and `bind 0.09 s`. A persistent
  plan cache (plans are rebuilt in every process) and module loading in the background are the planned reductions.
* Correctness: the whole CUDA suite (`ctest -L jit-app -R cuda`: laplace2d, poisson, lazy_triggers, CloverLeaf 2D and 3D, TeaLeaf, lowdim, TGV DP/SP/SPDP
  and HPSP, with and without MPI) passes on the A100 in verify mode (every group is compared with OPS's stock code).

## Compilation cost

Per group (local 16-thread WSL machine, LLVM -O3, host CPU features): `lowering` (MLIR passes) 3 ms, `llvm` (optimisation and code generation) 0.2 to 0.25 s;
CloverLeaf 2D compiles 65 distinct groups on a cold cache (15 to 17 s in total), 3D 95 groups (5 s for a 64^3 case, the groups are shared between
loops of equal structure). With a disk cache hit a group costs about 3 ms (`cache io`: 40 ms for 65 groups). On the GPU the lowering includes ptxas:
CloverLeaf 3D, 94 groups, cold: lowering 4.6 s, LLVM 3.6 s; warm: cache io 0.19 s.

## Reproducing

```bash
cluster/v2/sync.sh                                   # local branches -> the cluster work trees
sbatch cluster/v2/build.sbatch all                   # OPS, ops-mlir, the applications
sbatch cluster/v2/perf.sbatch cpu                    # CPU table (CASES=regex, SKIP_AOT=1, THREADS, NRUNS)
sbatch -p gpu --gres=gpu:a100:1 cluster/v2/perf.sbatch gpu
sbatch cluster/v2/perf.sbatch cpu-kernels            # per-kernel AOT against JIT (DIM=2|3)
sbatch cluster/v2/perf.sbatch cpu-groups             # time per JIT group (OPS_JIT_TRACE) and executor statistics
```
