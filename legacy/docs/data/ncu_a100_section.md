
## 7. Kernel-level view (Nsight Compute, A100)

One Runge-Kutta stage of the single-precision app at 128³, profiled with Nsight Compute
(`cluster/job_ncu.sbatch`, summarised by `ncu_summary.py`; raw CSVs in `data/`). "Occupancy" is
achieved warp occupancy, "DRAM util" the fraction of peak DRAM throughput during the kernel.

**Unfused (17 kernels)**

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

**Default planner (6 kernels)**

| kernel | µs | regs/thread | achieved occupancy | DRAM util | DRAM MB | L2 hit |
|---|---:|---:|---:|---:|---:|---:|
| `group_0` | 109 | 22 | 89% | 50% | 86 | 70% |
| `Kernel007_5` | 29 | 20 | 50% | 22% | 10 | 62% |
| `Kernel009_6` | 29 | 20 | 51% | 22% | 10 | 62% |
| `group_3` | 50 | 32 | 84% | 45% | 35 | 76% |
| `group_4` | 479 | 160 | 17% | 30% | 223 | 82% |
| `Kernel040_16` | 217 | 45 | 56% | 63% | 212 | 58% |
| **stage total** | **914** | 100 (time-weighted) | 41% | 41% | **576** | 73% |

Stage time 1267 → 914 µs (1.39×); DRAM traffic 854 → 576 MB (1.48×).


What the numbers show:

* The first fused kernel (`group_0`, five loops) is the ideal case: 22 registers, 89% occupancy, and it takes 109 µs where its five unfused parts took 198 µs.
* The largest fused kernel (`group_4`: loops 013, 016, 017, 018, 031, 032) moves 223 MB where its parts moved 443 MB (2.0× less), but it needs **160 registers per thread**, so its achieved occupancy is only 17% and it reaches 30% of peak DRAM throughput. It takes 479 µs against 707 µs for the unfused parts: 1.48× faster for 2.0× less traffic.
* `Kernel040` (the Runge-Kutta update, 212 MB) is untouched by fusion and is now the second-largest cost at 62% DRAM utilisation.
* The larger and faster the memory system, the more threads in flight a kernel needs to use it. The register-heavy fused kernel is therefore the part of the stage that falls short of the traffic saving. This is consistent with the smaller gain here than on the laptop GPU (whose 192 GB/s needs far less concurrency); it was not tested by capping register use.
