//===- Passes.td - transformation passes of the ops dialect ----*- tablegen -*-===//

#ifndef OPS_MLIR_TRANSFORMS_PASSES
#define OPS_MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def MaterializeKernels : Pass<"ops-materialize-kernels", "::mlir::ModuleOp"> {
  let summary = "clone the body of each ops.kernel into the ops.par_loops that call it";
  let description = [{
    A `par_loop` with a `kernel` symbol and an empty body is the unmaterialised form the lifter produces. This
    pass fills in its body from the `ops.kernel` of that name:

    - the i-th accessor parameter of the kernel is the i-th dat operand's block argument, the i-th reduction
      parameter the i-th reduction's;
    - a global parameter `!ops.gbl<T, n>` takes the next n scalar operands: `ops.gbl_read %g[c]` becomes the
      scalar block argument of component c;
    - `ops.const_ref` without an index becomes a new scalar argument of the chain (one per constant, offset
      and type), passed to every loop that uses it; the arguments are recorded in the chain attribute
      `ops.binding_consts`, an array of `{name, offset, arg}` the lifter reads back to bind them. A
      `const_ref` with a dynamic index stays and is handled by `ops-specialize-consts`.

    The result is verified: a read or write at an offset outside the operand's stencil is an error.
  }];
  let dependentDialects = ["::ops_mlir::ops::OPSDialect"];
}

def SpecializeConsts : Pass<"ops-specialize-consts", "::mlir::ModuleOp"> {
  let summary = "fold constants whose values are known, and unroll the loops that index them";
  let description = [{
    The values of constants come from the chain attribute `ops.const_values`, a dictionary from the name of
    a constant to its bytes (`array<i8: ...>`, little endian, as `ops_decl_const` registered it). Which
    constants to fold: the option `names`, by default every constant whose entry in the chain attribute
    `ops.const_roles` is `"bound"`.

    - arguments recorded in `ops.binding_consts` for such a constant are replaced by their value in every loop and
      removed from the chain signature;
    - `scf.for` loops with a constant trip count of at most `max-trip-count` that index such a constant with a
      dynamic `ops.const_ref` are fully unrolled;
    - `ops.const_ref` of such a constant with a constant (or no) index is replaced by its value;
    - canonicalisation then folds what became constant.

    A `ops.const_ref` with a dynamic index that remains is an error: a chain with such a reference cannot be
    compiled and the executor declines it.
  }];
  let options = [
    ListOption<"names", "names", "std::string", "constants to fold (default: those with role bound)">,
    Option<"maxTripCount", "max-trip-count", "int64_t", /*default=*/"256",
           "largest constant trip count of a loop that is fully unrolled">
  ];
  let dependentDialects = ["::ops_mlir::ops::OPSDialect", "::mlir::arith::ArithDialect", "::mlir::scf::SCFDialect"];
}

#endif // OPS_MLIR_TRANSFORMS_PASSES
