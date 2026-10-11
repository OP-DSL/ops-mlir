# ClangIR spike (plan phase 5): can frontend B replace the AST walker?

Question: is a frontend that reads **ClangIR** (`clang -fclangir -emit-cir`) a better foundation for importing OPS user kernels than the clang AST
walker of frontend A (`ops/lib/Frontend`, 1.8k lines: the C subset in `CSubsetWalker.cpp`, the OPS rules in `KernelImport.cpp`)? The criteria of the plan:
coverage at least that of A, no fidelity failures, at most 1.5k lines of raising code, and a measure of churn.

## 1. What was built and measured

* **A ClangIR-enabled clang of the pinned LLVM 23.1.0 sources** (`cluster/v2/build_cir.sbatch`: `-DCLANG_ENABLE_CIR=ON`, clang and MLIR only, 2.5 GB build
  tree, no install). **The pinned toolchain has none:** `clang -fclangir` answers "clang IR support not available, rebuild clang with -DCLANG_ENABLE_CIR=ON".
  Frontend B therefore means a second LLVM build for everybody who builds the applications (A uses the clang that every LLVM build has).
* **Corpus run** (`scripts/cir/corpus.py`, cluster): every translation unit that defines kernels, of every application tree of `apps/` (CloverLeaf 2D/3D,
  TeaLeaf, laplace2d, poisson, lazy_triggers, lowdim, TGV DP/SP/SPDP/HP/HPSP; the legacy `ops_seq.h` replaced by `ops_seq_v2.h`, `-DOPS_ACC_IGNORE`).
  CIR text parsed, the kernel functions found by name, their operations counted.

## 2. Results

| | |
|---|---|
| translation units | 62 |
| emitted CIR | **38 (61%)**; the 24 failures are all `use of undeclared identifier 'OPS_ACC0'` (and `OPS_ACC1..4`): the pointer-style kernels of CloverLeaf 3D use the macros `OPS_ACCn(i,j,k)` that the translator defines per loop (A has a rule for them: the same macro trick); not a CIR limitation |
| kernel functions found in the CIR | 1852 (counted per translation unit; 992 distinct per application) of the 2040 kernel entries of the corpus (the rest are in the 24 failed units) |
| unsupported-feature ("NYI") messages | 0 |
| operations per kernel function | minimum 9, median 63, maximum 1014 |

**The operations of the 992 kernels** (count): `cir.const` 59222, `cir.load` 32032, `cir.cast` 17493, `cir.call` 15740, `cir.store` 12036, `cir.fmul` 10928,
`cir.alloca` 7863, `cir.fdiv` 5144, `cir.fadd` 4628, `cir.get_global` 2477, `cir.fsub` 2258, `cir.fneg` 1910, `cir.return` 1852, `cir.ptr_stride` 729,
`cir.cmp` 225, `cir.scope` 186, `cir.if` 172, `cir.add` 123, `cir.yield` 110, `cir.fabs` 85, `cir.get_element` 85, `cir.mul` 65, `cir.div` 58,
`cir.get_member` 57, `cir.rem` 53, `cir.ternary` 48, `cir.sub` 23, `cir.for` 7, `cir.condition` 7, `cir.inc` 7, `cir.select` 4, `cir.fmaxnum` 1.
**33 operation kinds**; control flow is almost absent (`cir.if` 172, `cir.ternary` 48, `cir.for` 7 in 992 kernels).

**The calls** (15740): all but a few are `ACC<T>::operator()` (`_ZNK3ACCIdEclEiii`, `..IfE..`, `..IDF16_E..` const and non-const, 2-D and 3-D), plus
`ACC<T>::combine_{inc,min,max}` (6 each), `cos`/`sin`/`sqrt`/`exp`, and one helper function (`myfun`).

What the CIR of a kernel looks like (`calc_dt_kernel_min`: `*dt_min = *dt_min < dt_min_val(0,0) ? *dt_min : dt_min_val(0,0)`):

```
%0 = cir.alloca "dt_min" ... : !cir.ptr<!cir.ptr<!rec_ACC<double>>>        // one alloca per parameter, holding the argument
%1 = cir.alloca "dt_min_val" ...
cir.store %arg0, %0 ; cir.store %arg1, %1
%3 = cir.load %2 : the value *dt_min
%7 = cir.call @_ZNK3ACCIdEclEii(%4, %5, %6)       // ACC<double>::operator()(0, 0): the offsets are cir.const operands
%8 = cir.load %7
%9 = cir.cmp lt %3, %8
%10 = cir.ternary(%9, true { ... cir.yield %13 }, false { ... cir.yield %16 })
cir.store %10, %11
```

## 3. What a CIR raiser would have to do

| CIR | raised to |
|---|---|
| `alloca` of a parameter + the `store` of the argument, `load` of it | the parameter itself (accessor / global pointer) |
| `cir.call @...ACC...operator()(acc, i, j[, k])` + `cir.load` of the returned pointer | `ops.read %acc[i, j]` (offsets must be constants; unrolling as in A when a loop variable selects the offset) |
| the same call + `cir.store` to the returned pointer | `ops.write` (read-modify-write forms: read then write) |
| `cir.ptr_stride`/`get_element` on a global pointer parameter, `load`/`store` | `ops.gbl_read` / `ops.reduce`, with the same pattern recognition as A for `*r += v`, `*r = v` and the test-then-assign minimum |
| `cir.get_global` + `get_member` + `load` | `ops.const_ref` with the byte offset of the member (from the CIR struct layout) |
| scalar `alloca`/`load`/`store` | SSA values, threaded through `scf.if`/`scf.for` (CIR's own mem2reg works on CFG blocks, not on its structured regions) |
| `cir.const`, `cir.fadd` ... `cir.cmp`, `cir.cast`, `cir.fabs`, `cir.select`, `cir.ternary`, `cir.if`, `cir.for` | `arith`, `math`, `scf` |
| libm `cir.call` | `math` |
| `printf` calls | `ops.print` |

The work is almost all in the first four rows: **the OPS-specific recognition is the same as A's**, and so are the scalar SSA construction and the
loop policy. B would replace the *C parsing* (about 800 of A's 1800 lines: expression walking, casts, declarations, the type rules) by CIR
operations that are already typed and explicit (no implicit conversions, no integer promotions, no overload resolution, no macro or template
issues): an estimate of 600-900 lines for the raiser plus what A's `KernelImport.cpp` keeps (manifest, parameter classification, constants,
traits: about 900 lines, shared).

## 4. Assessment

| | Frontend A (AST walker) | Frontend B (ClangIR) |
|---|---|---|
| status | complete: 588 of 626 kernels of all OPS applications (`coverage.md`), every kernel of CloverLeaf, TeaLeaf, TGV; verify-clean on CPU, GPU, MPI | corpus and design only: **no raiser written** |
| toolchain | any clang 23 (the pinned build) | a clang built with `CLANG_ENABLE_CIR=ON`: a second LLVM build (the pinned one has none) |
| language coverage | the C subset of kernels; explicit "unsupported" diagnostics with the line | whatever clang's CIR generator supports (no NYI in the corpus; ClangIR is still under development: C++ constructs outside the corpus may hit NYI) |
| fragility | follows the clang AST API and the source forms of kernels (initialiser lists and `combine_*` calls are open, 38 kernels) | follows the CIR dialect, which is upstream but experimental and still gaining operations and changing attributes between releases (not measured here: a second LLVM version was not built) |
| OPS rules (accessors, reductions, constants) | in the walker | the same rules, on CIR patterns |
| size | 1.8k lines | est. 1.5k lines (raiser 600-900 + shared manifest 900) |

**Recommendation: keep A as the importer; do not build B now.** B does not add coverage (the open kernels of A are initialiser lists, accessor
`combine_*` calls, small integer and complex types, computed component indices: all expressible in A with a rule each, none a limitation of the
approach) and costs a second LLVM build plus a dialect that is still moving. It pays off only if (1) kernels in constructs the AST walker cannot
express become common (templates, lambdas, user-defined types in kernel bodies), or (2) the clang AST API churn costs more per release than CIR's would. The corpus
shows that CIR for the kernels of today is small and regular, so B stays a measured option, not a risk: the raiser can be written when one of those
two conditions holds, in a few days, against `scripts/cir/corpus.py` and the existing lit tests (`test/lit/Frontend`), which fix the expected `ops.kernel` of
each kernel shape.

To repeat the corpus run: `cluster/v2/build_cir.sbatch` (about 3 hours on 32 cores), then
`scripts/cir/corpus.py --clang ~/ops-jit-cl/build/llvm-cir/bin/clang --ops-include <OPS install>/include <apps build tree>...`.
