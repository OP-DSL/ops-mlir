//===- Passes.td - conversion passes of the ops dialect --------*- tablegen -*-===//

#ifndef OPS_MLIR_CONVERSION_PASSES
#define OPS_MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertOpsToSCF : Pass<"convert-ops-to-scf", "::mlir::ModuleOp"> {
  let summary = "lower a group (an ops.chain of one ops.par_loop) to a func.func with scf.parallel";
  let description = [{
    The compiled group ABI (plan section 4.6). The function takes, in this order:

    1. one memref per distinct dat of the loop, in order of first appearance;
    2. the shifts: one `index` per dat and dimension (dat-major, dimension-minor, OPS axis order);
    3. the range: lo, hi per dimension (OPS axis order, x first);
    4. the displacement of the block, one `index` per dimension (`ops.index d` is `i_d + disp_d`);
    5. the scalars of the loop (distinct chain arguments, in order of first appearance).

    It returns the values of the reductions (neutral element combined over the range). A dat of an OPS block of
    rank r with c components is a `memref<?x..xT>` of the r dimensions in memory order (z, y, x: OPS dimension d is
    memref dimension r-1-d) with a trailing dimension of size c if c > 1. Array of structures (default): the strides
    are [?, .., c, 1] (or [?, .., 1] for c = 1); structure of arrays (`soa`): [?, .., 1, ?]. An access at offset o of
    a dat with stencil stride s along d reads the memref index `i_d * s + o_d + shift_d`.

    The function is named like the chain, or as the chain attribute `ops.func_name` says; a private declaration of
    that name in the module (for a caller) is replaced by the definition.

    The function is marked `llvm.emit_c_interface`, its memref arguments `llvm.noalias`, and carries the attribute
    `ops.abi` describing the argument groups. Chains that contain `ops.call`s, and chains without a loop, are left
    alone. A group called from another chain in the same module cannot be converted (its signature changes): split
    the groups into modules of their own first.
  }];
  let options = [Option<"soa", "soa", "bool", /*default=*/"false", "structure-of-arrays layout of multi-component dats">];
  let dependentDialects = ["::ops_mlir::ops::OPSDialect", "::mlir::arith::ArithDialect", "::mlir::func::FuncDialect",
                           "::mlir::memref::MemRefDialect", "::mlir::scf::SCFDialect"];
}

#endif // OPS_MLIR_CONVERSION_PASSES
