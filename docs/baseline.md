# Baselines

What the OPS build looked like before the JIT, and what the changes to OPS (branch `feature/jit-executor`) did to it. Results live on the
cluster in `~/ops-jit-cl/results/*.txt` (one line per test) and are produced by `scripts/run_ops_ctest.sh`.

| Run | Tree | Platform | Tests | Passed | Notes |
|---|---|---|---|---|---|
| `ops-base-cpu` | OPS `develop` (before the changes) | CPU node, MPI np=2..4 | 73 | 73 | |
| `ops-new-cpu`, `-cpu2` | `feature/jit-executor` | CPU node | 89 | 83 -> 89 | the first run's failures were `lazy_triggers` (a shared output file: fixed in the test) and an MPI tiling limitation with loops that are empty on some ranks (the test no longer uses such a loop); `lowdim_*` once (concurrent HDF5 writes in parallel ctest: pass when serial) |
| `ops-base-a100` | `develop` | A100 | 31 | 27 | 4 failures that exist without the changes: `CloverLeaf_3D_HDF5` (CUDA variants: an HDF5 exception) and one flaky `laplace2d_cuda` (passes when run alone: parallel ctest on one GPU) |
| `ops-new-a100`, `-serial` | `feature/jit-executor` | A100 | 34 | 34 (the serial rerun of the flakes passes) | |

The OPS side changes are additive: with no executor registered the queueing code paths are the old ones, and the 89+34 tests above are
the existing OPS tests, unchanged except `lazy_triggers` (new). `OPS_LAZY_EXECUTOR_TEST=split|manual` runs any application through the
executor interface without the JIT to test that interface on its own.

The JIT tests (`ctest -L jit-app` in the build tree of `scripts/build_apps.sh`) are separate: see `performance.md` and `PROGRESS.md`.
The pre-existing failure (`CloverLeaf_3D_HDF5` on CUDA) is outside the JIT: the application is not built with it.
