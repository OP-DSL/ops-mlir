"""
Lower ops.par_loop -> func.func + stencil.* using xDSL, mirroring
lib/passes/OPSToStencil.cpp's C++ design.
"""

import ctypes
from dataclasses import dataclass
from xdsl.builder import Builder, InsertPoint
from xdsl.context import Context
from xdsl.dialects import arith, func, memref, stencil
from xdsl.dialects.builtin import IndexType, IntegerAttr, MemRefType, ModuleOp, i32, f64
from xdsl.ir import Block, Region, SSAValue
from xdsl.passes import ModulePass

from ops_dialect import ArgType, Access, DatAttr, ParLoopOp, StencilAttr

def field_bounds(dat: DatAttr) -> list[tuple[int, int]]:
    """Normalized 0-based field bounds: lb=0, ub=full allocated size per dim,
    reversed from OPS's [x, y, z] to put x last (unit-stride) -- see module note."""
    return [(0, size) for size in reversed(dat.size_list)]

def halo_offsets(dat_args) -> list[int]:
    """Per-dim d_m values (negative) from the first dat, reversed to match
    field_bounds's axis order -- shared across all dats on a block."""
    return list(reversed(dat_args[0].dat.d_m_list))

def stencil_offsets(stencil_attr: StencilAttr) -> list[tuple[int, ...]]:
    """Per-point access offsets for an arg's stencil.

    Extract the offsets from the StencilAttr's raw pointer to the underlying C array.
    """
    dims = stencil_attr.dims.data
    points = stencil_attr.points.data
    addr = stencil_attr.stencil.data
    if addr == 0 or points == 0:
        return [tuple(0 for _ in range(dims))]

    flat = (ctypes.c_int32 * (dims * points)).from_address(addr)
    return [tuple(reversed([flat[p * dims + d] for d in range(dims)])) for p in range(points)]

def range_bounds(rng: list[int], ndim: int) -> list[tuple[int, int]]:
    return list(reversed([(rng[2 * i], rng[2 * i + 1]) for i in range(ndim)]))

def normalized_range_bounds(rng: list[int], d_m: list[int], ndim: int) -> list[tuple[int, int]]:
    return [(lb - dm, ub - dm) for (lb, ub), dm in zip(range_bounds(rng, ndim), d_m)]

def declare_kernel(
    module: ModuleOp, name: str, num_f64_args: int, num_idx_args: int, ndim: int,
    num_results: int
) -> None:
    if any(
        isinstance(o, func.FuncOp) and o.sym_name.data == name
        for o in module.body.block.ops
    ):
        return

    param_types = (f64,) * num_f64_args + (MemRefType(i32, [ndim]),) * num_idx_args

    if num_results > 1:
        param_types = param_types + (MemRefType(f64, [num_results]),)
        result_types = ()
    else:
        result_types = (f64,) * max(num_results, 1)

    decl = func.FuncOp(
        name,
        (param_types, result_types),
        region=Region(),
        visibility="private",
    )
    module.body.block.add_op(decl)

def convert_par_loop(op: ParLoopOp, index: int, module: ModuleOp) -> func.FuncOp:
    ndim = op.dims.value.data
    args = op.arg_list()
    dat_args = [a for a in args if a.argtype.data == ArgType.DAT]

    # 
    gbl_args = [
        a for a in args
        if a.argtype.data == ArgType.GBL and a.acc.data == Access.READ
    ]
    num_idx_args = sum(1 for a in args if a.argtype.data == ArgType.IDX)

    for arg in gbl_args:
        if arg.dim.data != 1:
            raise NotImplementedError(
                f"ops_arg_gbl with dim={arg.dim.data} is not supported "
                f"(kernel '{op.kernel_name.data}'); only scalar (dim=1) "
                "double globals are modeled."
            )

    d_m = halo_offsets(dat_args)
    apply_bounds = stencil.StencilBoundsAttr(
        normalized_range_bounds(list(op.range.get_values()), d_m, ndim)
    )

    field_types = [stencil.FieldType(field_bounds(arg.dat), f64) for arg in dat_args]

    kernel_name = op.kernel_name.data
    fn_name = f"ops_par_loop_{kernel_name}_{index}"
    fn = func.FuncOp(
        fn_name, (tuple(field_types) + (f64,) * len(gbl_args), ()), visibility="private"
    )
    block = fn.body.block
    fn_builder = Builder(InsertPoint.at_end(block))
    dat_block_args = block.args[: len(dat_args)]
    gbl_block_args = list(block.args[len(dat_args) :])

    # Partition dat args (by access mode) into stencil.apply's reads/writes
    reads: list[SSAValue] = []
    read_types = []
    read_stencils: list[StencilAttr] = []
    writes: list[SSAValue] = []
    for arg, field in zip(dat_args, dat_block_args):
        access = arg.acc.data
        if access in (Access.READ, Access.RW):
            reads.append(field)
            read_types.append(field.type)
            read_stencils.append(arg.stencil)
        if access in (Access.WRITE, Access.RW, Access.INC):
            writes.append(field)

    apply_block = Block(arg_types=read_types + [f64] * len(gbl_args))
    block_builder = Builder(InsertPoint.at_end(apply_block))

    access_results = []
    for block_arg, stencil_attr in zip(apply_block.args[: len(read_types)], read_stencils):
        for point in stencil_offsets(stencil_attr):
            access = block_builder.insert(stencil.AccessOp(block_arg, point))
            access_results.append(access.res)

    gbl_values = list(apply_block.args[len(read_types) :])

    num_results = len(writes) or 1
    scope_block = Block()
    scope_builder = Builder(InsertPoint.at_end(scope_block))

    # Compute per-dim index buffers for the kernel call, which are passed as MemRefType(i32, [ndim])
    idx_buffers = []
    # Offset d_m[dim] un-normalizes the loop index back to the OPS global index:
    # ops_global_idx = normalized_loop_idx + d_m  (d_m is negative, e.g. -1)
    for _ in range(num_idx_args):
        idx_buffer = scope_builder.insert(
            memref.AllocaOp.get(i32, shape=[ndim])
        )
        for d in range(ndim):
            idx_op = scope_builder.insert(
                stencil.IndexOp.build(
                    attributes={
                        "dim": IntegerAttr(d, IndexType()),
                        "offset": stencil.IndexAttr.from_indices(*[
                            d_m[i] if i == d else 0 for i in range(ndim)
                        ]),
                    },
                    result_types=[IndexType()],
                )
            )
            idx_i32 = scope_builder.insert(
                arith.IndexCastOp(idx_op.idx, i32)
            )
            dim_const = scope_builder.insert(
                arith.ConstantOp(IntegerAttr(ndim - 1 - d, IndexType()))
            )
            scope_builder.insert(
                memref.StoreOp.get(idx_i32.result, idx_buffer.memref, [dim_const.result])
            )
        idx_buffers.append(idx_buffer.memref)

    call_args = access_results + gbl_values + idx_buffers
    declare_kernel(
        module, kernel_name, len(access_results) + len(gbl_args), len(idx_buffers),
        ndim, num_results
    )

    if num_results > 1:
        out_buf = scope_builder.insert(
            memref.AllocaOp.get(f64, shape=[num_results])
        )
        scope_builder.insert(
            func.CallOp(kernel_name, call_args + [out_buf.memref], [])
        )
        scope_results = []
        for i in range(num_results):
            idx_const = scope_builder.insert(arith.ConstantOp(IntegerAttr(i, IndexType())))
            loaded = scope_builder.insert(memref.LoadOp.get(out_buf.memref, [idx_const.result]))
            scope_results.append(loaded.res)
    else:
        kernel = scope_builder.insert(
            func.CallOp(kernel_name, call_args, [f64] * num_results)
        )
        scope_results = list(kernel.results)

    scope_builder.insert(memref.AllocaScopeReturnOp.build(operands=[scope_results]))

    alloca_scope = block_builder.insert(
        memref.AllocaScopeOp.build(
            regions=[Region([scope_block])],
            result_types=[[f64] * num_results],
        )
    )
    block_builder.insert(stencil.ReturnOp.get(list(alloca_scope.res)))

    fn_builder.insert(
        stencil.ApplyOp.build(
            operands=[reads + gbl_block_args, writes, []],  # reduction operands empty for now
            regions=[Region([apply_block])],
            result_types=[[]],  # buffer semantic does not return results
            properties={"bounds": apply_bounds},
        )
    )

    fn_builder.insert(func.ReturnOp())
    return fn

@dataclass(frozen=True)
class OPSToStencilPass(ModulePass):
    """Lower ops.par_loop -> func.func + stencil.* (see module docstring)."""
    name = "ops-to-stencil"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        loops = [o for o in op.body.block.ops if isinstance(o, ParLoopOp)]
        for index, loop_op in enumerate(loops):
            self.lower_par_loop(loop_op, index, op)

    def lower_par_loop(self, loop_op: ParLoopOp, index: int, module: ModuleOp) -> None:
        fn = convert_par_loop(loop_op, index, module)
        module.body.block.add_op(fn)
        loop_op.detach()
        loop_op.erase()
