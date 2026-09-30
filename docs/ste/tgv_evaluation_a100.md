# Taylor-Green vortex: fusion and reordering evaluation - NVIDIA A100-PCIE-40GB

> This is the Simplified Technical English (ASD-STE100) version of [tgv_evaluation_a100.md](../tgv_evaluation_a100.md). Code, listings and numbers are the same as in the original.

`apps/c/taylor_green_vortex/eval_tgv.py` made this document on 2026-10-01 09:37:50. The GPU is an NVIDIA A100-PCIE-40GB, 40960 MiB, 580.95.05. The host has 48 CPU threads. The raw data is in `data/tgv_eval_a100.json`.

## Summary

* **One Runge-Kutta stage** queues 17 loops. Fusion of adjacent loops only puts them in 8 generated kernels. The planner with the dependence DAG puts them in 6 generated kernels and moves 3 loops past independent loops. Launches per iteration (with the filter steps): 51 → 24 → 18.
* **Single precision on NVIDIA A100-PCIE-40GB (256³):** 24.19 ms/iteration unfused → 20.50 ms with adjacent same-range fusion (1.18×) → 19.96 ms with the DAG planner (1.21×). Ordinary fusion gives most of the gain. Reordering adds 3% more. Different-range (guarded) fusion changes nothing for this application: the stage has the same number of generated kernels with or without it.
* **Why the gain is less than the traffic saving:** The byte model predicts 1.54× less traffic, but the kernel time drops only 1.23×. The modelled bandwidth falls from 911 GB/s (59% of the 1555 GB/s peak) unfused to 724 GB/s (47%) fused. The fused generated kernels move fewer bytes, but they reach less bandwidth (see the Nsight Compute section for the cause in each GPU kernel). The host overhead (enqueue, planning, key, argument packing) is less than 2% of an iteration. So the number of launches does not cost time by itself.
* **Double precision on NVIDIA A100-PCIE-40GB:** 54.43 → 51.86 ms (1.05×). Double precision costs 2.6× single precision. Twice the bytes alone gives 2×, so the cost is somewhat more than that. Fusion helps double precision much less than single precision (1.05× against 1.21×).
* **OpenMP (24 threads, 64³):** 1.13× in single precision (40.33 → 35.66 ms) and 1.08× in double precision (40.80 → 37.75 ms). Section 2 gives every configuration and its spread.
* **Problem size (single precision, GPU):** For N = 32…384, the speed-up is from 1.07× (N=320) to 1.34× (N=128). It does not change in one direction when N changes. No one investigated the dips.
* **Parameters:** A cap of 8 loops for each generated kernel already reaches the best time (6 generated kernels for each stage). Larger caps change nothing, and a cap of 1 is the unfused case. The bounding-box ratio has little effect here (19.94–19.96 ms). Earliest placement and latest placement: 19.96 ms and 19.97 ms.
* **Cost:** The JIT compilation of the 9 different queue shapes takes 11.4 s unfused and 10.3 s fused. The first 100 iterations take about 9.0 s more than steady state. This cost is a one-off. It equals about 514 steady iterations.
* **Correctness:** All 20 runs of a planner configuration are bitwise identical to the unfused run (CPU and GPU, both precisions, see section 6). After 26 steps, single precision stays within 2e-06 of double precision (rms, relative to the flow scale).
* **What limits it:** A stage still needs many generated kernels. Some loops had to start a new generated kernel. For these loops, the diagnostics of the planner count the reasons why the planner rejected an existing generated kernel: `dependence` 5, `order` 1, `range` 3.
  * `dependence`: the loop has a stencil read of a field that a member of that generated kernel writes.
  * `order`: if the loop joins, the planner must move the loop before a loop that it depends on.
  * `range`: the rules of the box reject the loop.
  * Fusion across the dependence edges needs redundant halo computation.

*Method note.* The times for each iteration come from windows of 100 iterations in one process after the warm-up. An earlier version of this evaluation used the difference of the wall time of a short run and a long run. The JIT time changes by about 1 s between runs, which is ±10 ms for each iteration. That method gave wrong numbers, and it also counted the one-off compile of the filter generated kernels at iteration 25. The authors discarded those numbers.


## 1. What the planner does to one time step

One Runge-Kutta stage queues a batch of loops between halo exchanges. A time step has three stages. A filter block runs every 25 steps. The filter block has seven loops that the application flushes one by one.

| configuration | loops / stage | generated kernels / stage | loops moved | est. traffic (unfused → fused) | reduction |
|---|---:|---:|---:|---:|---:|
| unfused | 17 | 17 | 0 | 4.11 → 4.11 MB (N=16) | 1.00× |
| adjacent, same-range | 17 | 8 | 0 | 4.11 → 2.74 MB (N=16) | 1.50× |
| adjacent, guarded | 17 | 8 | 0 | 4.11 → 2.74 MB (N=16) | 1.50× |
| DAG, same-range | 17 | 6 | 3 | 4.11 → 2.63 MB (N=16) | 1.56× |
| DAG, guarded | 17 | 6 | 3 | 4.11 → 2.63 MB (N=16) | 1.56× |
| DAG, guarded, latest | 17 | 6 | 3 | 4.11 → 2.63 MB (N=16) | 1.56× |

Each time step makes 51 launches unfused and 18 launches with the default planner. The filter block is not in these counts.

Why loops still start a new generated kernel: the counts are the rejected candidate generated kernels for the default planner and one stage. They are `dependence`=5, `first`=1, `order`=1, `range`=3.

## 2. Time per iteration

The steady-state time per iteration is the timing that the application itself gives for windows of 100 iterations in one process. The windows are **after** the warm-up window. The warm-up window contains the JIT compile of every generated kernel shape.

The values are medians over all steady windows of all repetitions. `±` is (max−min)/median over those windows. `warm-up` is the extra time that the first 100 iterations took, compared with steady state. `JIT` is the total seconds that the runtime spent to compile.

### single precision, CUDA, 256³

| configuration | ms / iteration | ± | speed-up vs unfused | generated kernels / iteration | warm-up s | JIT s |
|---|---:|---:|---:|---:|---:|---:|
| unfused | 24.19 | 0.8% | 1.000× | 51.3 | 10.2 | 11.4 |
| adjacent, same-range | 20.50 | 0.6% | 1.180× | 24.3 | 9.2 | 10.4 |
| adjacent, guarded | 20.54 | 0.4% | 1.178× | 24.3 | 9.2 | 10.4 |
| DAG, same-range | 19.94 | 0.4% | 1.213× | 18.3 | 9.0 | 10.3 |
| DAG, guarded | 19.96 | 0.3% | 1.212× | 18.3 | 9.0 | 10.3 |
| DAG, guarded, latest | 19.97 | 0.3% | 1.211× | 18.3 | 9.0 | 10.3 |

![opensbli_f32/cuda/N256](../img/timing_opensbli_f32_cuda_N256_a100.svg)

### double precision, CUDA, 256³

| configuration | ms / iteration | ± | speed-up vs unfused | generated kernels / iteration | warm-up s | JIT s |
|---|---:|---:|---:|---:|---:|---:|
| unfused | 54.43 | 0.2% | 1.000× | 51.3 | 10.5 | 11.3 |
| adjacent, same-range | 51.91 | 0.1% | 1.049× | 24.3 | 9.6 | 10.3 |
| adjacent, guarded | 51.93 | 0.1% | 1.048× | 24.3 | 9.5 | 10.3 |
| DAG, same-range | 51.87 | 0.1% | 1.049× | 18.3 | 9.4 | 10.2 |
| DAG, guarded | 51.86 | 0.1% | 1.050× | 18.3 | 9.4 | 10.2 |
| DAG, guarded, latest | 51.86 | 0.1% | 1.049× | 18.3 | 9.4 | 10.2 |

![opensbli/cuda/N256](../img/timing_opensbli_cuda_N256_a100.svg)

### single precision, OPENMP, 64³

| configuration | ms / iteration | ± | speed-up vs unfused | generated kernels / iteration | warm-up s | JIT s |
|---|---:|---:|---:|---:|---:|---:|
| unfused | 40.02 | 3.8% | 1.000× | 51.3 | 8.9 | 9.3 |
| adjacent, same-range | 35.05 | 3.6% | 1.142× | 24.3 | 8.1 | 8.6 |
| adjacent, guarded | 35.33 | 4.9% | 1.133× | 24.3 | 8.2 | 8.6 |
| DAG, same-range | 35.03 | 7.4% | 1.143× | 18.3 | 8.1 | 8.6 |
| DAG, guarded | 35.52 | 4.9% | 1.127× | 18.3 | 8.0 | 8.5 |
| DAG, guarded, latest | 35.26 | 3.6% | 1.135× | 18.3 | 8.1 | 8.6 |

![opensbli_f32/openmp/N64](../img/timing_opensbli_f32_openmp_N64_a100.svg)

### double precision, OPENMP, 64³

| configuration | ms / iteration | ± | speed-up vs unfused | generated kernels / iteration | warm-up s | JIT s |
|---|---:|---:|---:|---:|---:|---:|
| unfused | 40.66 | 5.8% | 1.000× | 51.3 | 8.9 | 9.4 |
| adjacent, same-range | 37.40 | 7.8% | 1.087× | 24.3 | 8.0 | 8.5 |
| adjacent, guarded | 36.66 | 3.5% | 1.109× | 24.3 | 8.1 | 8.6 |
| DAG, same-range | 37.16 | 3.1% | 1.094× | 18.3 | 8.0 | 8.5 |
| DAG, guarded | 37.55 | 3.3% | 1.083× | 18.3 | 8.0 | 8.5 |
| DAG, guarded, latest | 37.38 | 6.0% | 1.088× | 18.3 | 8.0 | 8.4 |

![opensbli/openmp/N64](../img/timing_opensbli_openmp_N64_a100.svg)

### OpenMP, controlled repeat (5 repetitions)

The OpenMP runs above use every available thread and can have noise. The runs below use a fixed number of threads and more repetitions.

**single precision, 64³, 24 threads**

| configuration | ms / iteration | ± | speed-up vs unfused |
|---|---:|---:|---:|
| unfused | 40.33 | 4.6% | 1.000× |
| adjacent, same-range | 34.69 | 2.0% | 1.163× |
| adjacent, guarded | 35.76 | 3.6% | 1.128× |
| DAG, guarded | 35.66 | 4.8% | 1.131× |

**double precision, 64³, 24 threads**

| configuration | ms / iteration | ± | speed-up vs unfused |
|---|---:|---:|---:|
| unfused | 40.80 | 4.8% | 1.000× |
| adjacent, same-range | 37.45 | 4.0% | 1.090× |
| adjacent, guarded | 37.29 | 3.2% | 1.094× |
| DAG, guarded | 37.75 | 7.4% | 1.081× |

Single precision against double precision on the GPU (default planner, 256³): 19.96 ms and 51.86 ms per iteration. The ratio is **2.60×**.

## 3. Dependence on problem size (single precision, GPU)

| N³ | unfused ms | adjacent, same-range ms | DAG, guarded ms | speed-up (default) |
|---:|---:|---:|---:|---:|
| 32 | 1.39 | 1.11 | 1.16 | 1.198× |
| 64 | 1.64 | 1.30 | 1.36 | 1.201× |
| 96 | 2.50 | 1.93 | 1.90 | 1.311× |
| 128 | 4.15 | 3.22 | 3.09 | 1.345× |
| 192 | 11.12 | 8.70 | 8.39 | 1.325× |
| 256 | 24.28 | 20.54 | 19.96 | 1.217× |
| 320 | 48.61 | 46.57 | 45.31 | 1.073× |
| 384 | 94.63 | 80.69 | 78.62 | 1.204× |

![sizes](../img/sizes_a100.svg)

### Double precision (GPU)

| N³ | unfused ms | adjacent, same-range ms | DAG, guarded ms | speed-up (default) |
|---:|---:|---:|---:|---:|
| 64 | 1.94 | 1.59 | 1.64 | 1.185× |
| 128 | 7.04 | 5.81 | 5.75 | 1.225× |
| 192 | 20.76 | 20.33 | 20.56 | 1.010× |
| 256 | 54.42 | 51.92 | 51.85 | 1.050× |

## 4. Parameter sweeps (single precision, GPU, 256³)

Maximum loops for each generated kernel (`OPS_MLIR_FUSION_MAX`):

| max | generated kernels / stage | ms / iteration |
|---:|---:|---:|
| 1 | 17 | 24.54 |
| 2 | 11 | 22.68 |
| 4 | 8 | 21.85 |
| 8 | 6 | 19.98 |
| 16 | 6 | 19.95 |
| 32 | 6 | 19.97 |

Bounding-box ratio (`OPS_MLIR_FUSION_BOX_RATIO`):

| ratio | generated kernels / stage | ms / iteration |
|---:|---:|---:|
| 0.5 | 6 | 19.96 |
| 1.0 | 6 | 19.94 |
| 2.0 | 6 | 19.95 |
| 4.0 | 6 | 19.94 |

## 5. Where the time goes

This is the exact steady-state breakdown for each iteration (single precision, GPU, 256³). It comes from the cumulative timers of the runtime. Each value is the difference between windows of 100 iterations.

*kernel* is the time in the generated functions, with the device synchronisation. *run overhead* is the argument packing, the buffer bookkeeping and the profiling.

| configuration | wall ms | generated kernels | run overhead | halo exchange | enqueue | plan + key | other (application) |
|---|---:|---:|---:|---:|---:|---:|---:|
| unfused | 24.19 | 23.44 | 0.08 | 0.39 | 0.14 | 0.12 | 0.01 |
| adjacent, same-range | 20.50 | 19.72 | 0.07 | 0.39 | 0.14 | 0.17 | 0.00 |
| DAG, guarded | 19.96 | 19.09 | 0.07 | 0.39 | 0.15 | 0.26 | -0.00 |

Modelled memory bandwidth (the byte model of the generated kernels that the runtime launched × 3 stages ÷ kernel time). The peak of the GPU is 1555.0 GB/s.

| configuration | model traffic / iteration | kernel ms | modelled GB/s |
|---|---:|---:|---:|
| unfused | 21.36 GB | 23.44 | 911 |
| DAG, guarded | 13.83 GB | 19.09 | 724 |

Share of each generated kernel in the kernel time (profiler, 250 iterations):

**unfused**: total kernel time 1.239 s

| kernel | calls | total s | share |
|---|---:|---:|---:|
| `opensbliblock00Kernel008` | 750 | 0.308 | 25% |
| `opensbliblock00Kernel032` | 750 | 0.184 | 15% |
| `opensbliblock00Kernel031` | 750 | 0.168 | 14% |
| `opensbliblock00Kernel040` | 750 | 0.151 | 12% |
| `opensbliblock00Kernel019` | 750 | 0.043 | 3% |
| `opensbliblock00Kernel010` | 750 | 0.028 | 2% |

**DAG, guarded**: total kernel time 0.896 s

| kernel | calls | total s | share |
|---|---:|---:|---:|
| `fused(opensbliblock00Kernel008+opensbliblock00Kernel010+open` | 750 | 0.288 | 32% |
| `fused(opensbliblock00Kernel013+opensbliblock00Kernel016+open` | 750 | 0.265 | 30% |
| `opensbliblock00Kernel040` | 750 | 0.154 | 17% |
| `fused(opensbliblock00Kernel011+opensbliblock00Kernel014+open` | 750 | 0.037 | 4% |
| `opensbliblock00Kernel007` | 750 | 0.023 | 3% |
| `opensbliblock00Kernel009` | 750 | 0.022 | 2% |

## 6. Accuracy

The table compares each configuration with the unfused run (same backend and precision). The size is 32³ and each run has 26 steps. The steps include a filter step.

| application / backend | configuration | bitwise equal | max abs diff | rel L2 |
|---|---|:---:|---:|---:|
| opensbli/seq | adjacent, same-range | yes | 0.00e+00 | 0.00e+00 |
| opensbli/seq | adjacent, guarded | yes | 0.00e+00 | 0.00e+00 |
| opensbli/seq | DAG, same-range | yes | 0.00e+00 | 0.00e+00 |
| opensbli/seq | DAG, guarded | yes | 0.00e+00 | 0.00e+00 |
| opensbli/seq | DAG, guarded, latest | yes | 0.00e+00 | 0.00e+00 |
| opensbli_f32/seq | adjacent, same-range | yes | 0.00e+00 | 0.00e+00 |
| opensbli_f32/seq | adjacent, guarded | yes | 0.00e+00 | 0.00e+00 |
| opensbli_f32/seq | DAG, same-range | yes | 0.00e+00 | 0.00e+00 |
| opensbli_f32/seq | DAG, guarded | yes | 0.00e+00 | 0.00e+00 |
| opensbli_f32/seq | DAG, guarded, latest | yes | 0.00e+00 | 0.00e+00 |
| opensbli_f32/cuda | adjacent, same-range | yes | 0.00e+00 | 0.00e+00 |
| opensbli_f32/cuda | adjacent, guarded | yes | 0.00e+00 | 0.00e+00 |
| opensbli_f32/cuda | DAG, same-range | yes | 0.00e+00 | 0.00e+00 |
| opensbli_f32/cuda | DAG, guarded | yes | 0.00e+00 | 0.00e+00 |
| opensbli_f32/cuda | DAG, guarded, latest | yes | 0.00e+00 | 0.00e+00 |
| opensbli/cuda | adjacent, same-range | yes | 0.00e+00 | 0.00e+00 |
| opensbli/cuda | adjacent, guarded | yes | 0.00e+00 | 0.00e+00 |
| opensbli/cuda | DAG, same-range | yes | 0.00e+00 | 0.00e+00 |
| opensbli/cuda | DAG, guarded | yes | 0.00e+00 | 0.00e+00 |
| opensbli/cuda | DAG, guarded, latest | yes | 0.00e+00 | 0.00e+00 |

Single precision against double precision (rms error relative to the flow scale): rho 2.79e-07, rhou0 1.91e-06, rhou1 1.92e-06, rhou2 1.89e-06, rhoE 1.47e-07.


## 7. View of the GPU kernels (Nsight Compute, A100)

Nsight Compute profiled one Runge-Kutta stage of the single-precision application at 128³ (`cluster/job_ncu.sbatch`). `ncu_summary.py` made the summary. The raw CSV files are in `data/`. "Occupancy" is the achieved warp occupancy. "DRAM util" is the fraction of the peak DRAM throughput during the GPU kernel.

**Unfused (17 GPU kernels)**

| kernel | µs | regs/thread | achieved occupancy | DRAM util | DRAM MB | L2 hit |
|---|---:|---:|---:|---:|---:|---:|
| `Kernel008_0` | 36 | 16 | 64% | 41% | 23 | 53% |
| `Kernel010_1` | 36 | 16 | 57% | 42% | 23 | 53% |
| `Kernel012_2` | 39 | 16 | 74% | 38% | 23 | 51% |
| `Kernel019_3` | 52 | 18 | 85% | 70% | 56 | 46% |
| `Kernel025_4` | 35 | 16 | 56% | 42% | 23 | 53% |
| `Kernel007_5` | 29 | 20 | 50% | 22% | 10 | 62% |
| `Kernel009_6` | 28 | 20 | 50% | 22% | 10 | 62% |
| `Kernel011_7` | 29 | 20 | 50% | 22% | 10 | 60% |
| `Kernel013_8` | 28 | 22 | 61% | 24% | 10 | 67% |
| `Kernel014_9` | 29 | 22 | 67% | 23% | 10 | 67% |
| `Kernel015_10` | 29 | 22 | 67% | 23% | 10 | 65% |
| `Kernel016_11` | 33 | 18 | 78% | 19% | 10 | 75% |
| `Kernel017_12` | 33 | 18 | 78% | 20% | 10 | 75% |
| `Kernel018_13` | 33 | 18 | 78% | 19% | 10 | 74% |
| `Kernel031_14` | 286 | 119 | 23% | 47% | 210 | 71% |
| `Kernel032_15` | 294 | 56 | 53% | 42% | 193 | 72% |
| `Kernel040_16` | 219 | 45 | 56% | 62% | 212 | 58% |
| **stage total** | **1267** | 54 (time-weighted) | 52% | 43% | **854** | 65% |

**Default planner (6 GPU kernels)**

| kernel | µs | regs/thread | achieved occupancy | DRAM util | DRAM MB | L2 hit |
|---|---:|---:|---:|---:|---:|---:|
| `group_0` | 109 | 22 | 89% | 50% | 86 | 70% |
| `Kernel007_5` | 29 | 20 | 50% | 22% | 10 | 62% |
| `Kernel009_6` | 29 | 20 | 51% | 22% | 10 | 62% |
| `group_3` | 50 | 32 | 84% | 45% | 35 | 76% |
| `group_4` | 479 | 160 | 17% | 30% | 223 | 82% |
| `Kernel040_16` | 217 | 45 | 56% | 63% | 212 | 58% |
| **stage total** | **914** | 100 (time-weighted) | 41% | 41% | **576** | 73% |

The stage time goes from 1267 to 914 µs (1.39×). The DRAM traffic goes from 854 to 576 MB (1.48×).


What the numbers show:

* The first fused GPU kernel (`group_0`, five loops) is the ideal case. It uses 22 registers and has 89% occupancy. It takes 109 µs, but its five unfused parts took 198 µs.
* The largest fused GPU kernel (`group_4`: loops 013, 016, 017, 018, 031, 032) moves 223 MB, but its parts moved 443 MB (2.0× less). It needs **160 registers per thread**. So its achieved occupancy is only 17%, and it reaches 30% of the peak DRAM throughput. It takes 479 µs against 707 µs for the unfused parts. It is 1.48× faster for 2.0× less traffic.
* Fusion does not change `Kernel040` (the Runge-Kutta update, 212 MB). It is now the second-largest cost, with 62% DRAM utilisation.
* A larger and faster memory system needs more threads that run at the same time in a GPU kernel to use the system. Because of this, the fused GPU kernel with many registers is the part of the stage that does not reach the traffic saving. This agrees with the smaller gain here than on the laptop GPU. The laptop GPU has 192 GB/s and needs much less concurrency. No one tested this with a cap on the register use.
