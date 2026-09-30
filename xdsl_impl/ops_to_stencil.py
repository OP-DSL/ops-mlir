"""
Lower ops.par_loop -> func.func + stencil.* using xDSL, mirroring
lib/passes/OPSToStencil.cpp's C++ design.

Part of OPS-MLIR Project
Author: Prakanth Thilakaraj
Date: June 2026

This file is distributed under the MIT License.
See LICENSE.txt for details.
"""

import ctypes
from dataclasses import dataclass
from xdsl.builder import Builder, InsertPoint
from xdsl.context import Context
from xdsl.dialects import arith, func, memref, scf, stencil
from xdsl.dialects.builtin import IndexType, IntegerAttr, MemRefType, ModuleOp, i32, f32, f64
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

_ELT_BY_TYPE_NAME = {"float": f32, "double": f64}
_ELT_BY_BYTES = {4: f32, 8: f64}


def dat_elt(dat: DatAttr):
    """MLIR element type of a dat, from its OPS type string."""
    name = dat.dat_type.data
    if name not in _ELT_BY_TYPE_NAME:
        raise NotImplementedError(
            f"dat '{dat.dat_name.data}' has type '{name}'; only float and double "
            "are supported"
        )
    return _ELT_BY_TYPE_NAME[name]


def gbl_elt(arg):
    """MLIR element type of a scalar global. OPS ignores the type string of
    ops_arg_gbl and only records sizeof(T), so the size picks the type."""
    nbytes = arg.elem_size.data // max(arg.dim.data, 1)
    if nbytes not in _ELT_BY_BYTES:
        raise NotImplementedError(
            f"ops_arg_gbl with element size {nbytes} bytes is not supported; "
            "only float (4) and double (8) globals are modeled."
        )
    return _ELT_BY_BYTES[nbytes]


def declare_kernel(
    module: ModuleOp, name: str, value_types: list, num_idx_args: int, ndim: int,
    num_results: int, result_type
) -> None:
    if any(
        isinstance(o, func.FuncOp) and o.sym_name.data == name
        for o in module.body.block.ops
    ):
        return

    param_types = tuple(value_types) + (MemRefType(i32, [ndim]),) * num_idx_args

    if num_results > 1:
        param_types = param_types + (MemRefType(result_type, [num_results]),)
        result_types = ()
    else:
        result_types = (result_type,) * max(num_results, 1)

    decl = func.FuncOp(
        name,
        (param_types, result_types),
        region=Region(),
        visibility="private",
    )
    module.body.block.add_op(decl)

def _is_zero(point) -> bool:
    return all(c == 0 for c in point)


def group_func_name(loops: list[ParLoopOp], loop_indices: list[int], group_id: int) -> str:
    """Name of the generated function for a group. The C++ runtime derives the
    same name (JITEngine::groupFunctionName); keep the two in sync."""
    if len(loops) == 1:
        return f"ops_par_loop_{loops[0].kernel_name.data}_{loop_indices[0]}"
    return f"ops_par_loop_group_{group_id}"


def convert_group(
    loops: list[ParLoopOp], fn_name: str, module: ModuleOp
) -> func.FuncOp:
    """Lower one or more consecutive ops.par_loop into a single function with
    one stencil.apply. A group of one is the plain per-loop lowering.

    Function signature: one field per distinct dat (by OPS dat index, in first
    appearance order), then the read-only scalar globals of every member in
    order. The runtime (JITEngine::execute) packs arguments the same way.

    Members run in order at each point. A value written by an earlier member
    and read at the zero offset by a later member is forwarded as an SSA value
    instead of being re-loaded (the dat is still stored). The planner
    guarantees no member reads a dat non-locally after an earlier member wrote
    it, or writes a dat an earlier member read non-locally.
    """
    ndim = loops[0].dims.value.data

    # Distinct dats, first-appearance order.
    dat_keys: list[int] = []
    dat_attr: dict[int, DatAttr] = {}
    for op in loops:
        for a in op.arg_list():
            if a.argtype.data == ArgType.DAT:
                key = a.dat.index.data
                if key not in dat_attr:
                    dat_attr[key] = a.dat
                    dat_keys.append(key)

    # d_m is shared by all dats on a block (see halo_offsets).
    d_m = list(reversed(dat_attr[dat_keys[0]].d_m_list))

    # Bounding box of the members' (normalized) ranges.
    member_bounds = [
        normalized_range_bounds(list(op.range.get_values()), d_m, ndim) for op in loops
    ]
    box = [
        (min(b[d][0] for b in member_bounds), max(b[d][1] for b in member_bounds))
        for d in range(ndim)
    ]
    # A member whose range is smaller than the box is "guarded": the fused
    # kernel visits every point of the box, so that member's body only runs
    # where the point is inside its range (see the scf.if below); elsewhere the
    # dats it writes keep their previous value. The planner only forms such
    # groups when the member reads point-locally, so no out-of-range stencil
    # read can happen.
    guarded_members = [bounds != box for bounds in member_bounds]
    apply_bounds = stencil.StencilBoundsAttr(box)

    # Read-only scalar globals of every member, in order.
    gbl_args = []
    for op in loops:
        for a in op.arg_list():
            if a.argtype.data == ArgType.GBL and a.acc.data == Access.READ:
                if a.dim.data != 1:
                    raise NotImplementedError(
                        f"ops_arg_gbl with dim={a.dim.data} is not supported "
                        f"(kernel '{op.kernel_name.data}'); only scalar (dim=1) "
                        "globals are modeled."
                    )
                gbl_args.append(a)
    gbl_types = [gbl_elt(a) for a in gbl_args]

    field_types = {
        key: stencil.FieldType(field_bounds(dat_attr[key]), dat_elt(dat_attr[key]))
        for key in dat_keys
    }

    fn = func.FuncOp(
        fn_name,
        (tuple(field_types[k] for k in dat_keys) + tuple(gbl_types), ()),
        visibility="private",
    )
    block = fn.body.block
    fn_builder = Builder(InsertPoint.at_end(block))
    field_of = dict(zip(dat_keys, block.args[: len(dat_keys)]))
    gbl_block_args = list(block.args[len(dat_keys):])

    # Dats read (READ/RW) by any member, and dats written by any member, in
    # first-use order.
    read_keys: list[int] = []
    write_keys: list[int] = []
    for op, is_guarded in zip(loops, guarded_members):
        for a in op.arg_list():
            if a.argtype.data != ArgType.DAT:
                continue
            key = a.dat.index.data
            if a.acc.data in (Access.READ, Access.RW) and key not in read_keys:
                read_keys.append(key)
            if a.acc.data in (Access.WRITE, Access.RW, Access.INC):
                if key not in write_keys:
                    write_keys.append(key)
                # A guarded write keeps the old value outside its range, so
                # the dat's current contents must be readable.
                if is_guarded and key not in read_keys:
                    read_keys.append(key)

    apply_block = Block(
        arg_types=[field_of[k].type for k in read_keys] + gbl_types
    )
    block_builder = Builder(InsertPoint.at_end(apply_block))
    temp_of = dict(zip(read_keys, apply_block.args[: len(read_keys)]))
    gbl_values = list(apply_block.args[len(read_keys):])

    accesses: dict[tuple, SSAValue] = {}

    def access(key: int, point: tuple) -> SSAValue:
        memo = (key, point)
        if memo not in accesses:
            accesses[memo] = block_builder.insert(
                stencil.AccessOp(temp_of[key], point)
            ).res
        return accesses[memo]

    def in_range(bounds) -> SSAValue | None:
        """i1 that is true where the iteration point lies inside `bounds`
        (only dims narrower than the box are tested)."""
        cond = None
        for d, (lo, hi) in enumerate(bounds):
            if (lo, hi) == box[d]:
                continue
            ix = block_builder.insert(
                stencil.IndexOp.build(
                    attributes={
                        "dim": IntegerAttr(d, IndexType()),
                        "offset": stencil.IndexAttr.from_indices(*[0] * ndim),
                    },
                    result_types=[IndexType()],
                )
            ).idx
            lo_c = block_builder.insert(arith.ConstantOp(IntegerAttr(lo, IndexType()))).result
            hi_c = block_builder.insert(arith.ConstantOp(IntegerAttr(hi, IndexType()))).result
            ge = block_builder.insert(arith.CmpiOp(ix, lo_c, "sge")).result
            lt = block_builder.insert(arith.CmpiOp(ix, hi_c, "slt")).result
            both = block_builder.insert(arith.AndIOp(ge, lt)).result
            cond = both if cond is None else block_builder.insert(arith.AndIOp(cond, both)).result
        return cond

    current: dict[int, SSAValue] = {}  # latest value written per dat
    last_results: list[SSAValue] = []
    gbl_cursor = 0

    for member, op in enumerate(loops):
        args = op.arg_list()
        dat_args = [a for a in args if a.argtype.data == ArgType.DAT]
        num_idx_args = sum(1 for a in args if a.argtype.data == ArgType.IDX)
        kernel_name = op.kernel_name.data

        access_results = []
        for a in dat_args:
            if a.acc.data not in (Access.READ, Access.RW):
                continue
            key = a.dat.index.data
            for point in stencil_offsets(a.stencil):
                if _is_zero(point) and key in current:
                    access_results.append(current[key])  # forwarded
                else:
                    access_results.append(access(key, point))

        n_gbl = sum(
            1 for a in args
            if a.argtype.data == ArgType.GBL and a.acc.data == Access.READ
        )
        member_gbl = gbl_values[gbl_cursor: gbl_cursor + n_gbl]
        gbl_cursor += n_gbl

        written = [
            a for a in dat_args
            if a.acc.data in (Access.WRITE, Access.RW, Access.INC)
        ]
        write_elts = {dat_elt(a.dat) for a in written}
        if len(write_elts) > 1:
            raise NotImplementedError(
                f"kernel '{kernel_name}' writes dats of different element types; "
                "all written dats of one loop must share a type."
            )
        result_elt = next(iter(write_elts)) if write_elts else f64
        num_results = len(written) or 1

        # Guarded member: run the kernel only where the point is inside the
        # member's range; elsewhere keep the dats' previous values.
        cond = in_range(member_bounds[member]) if (guarded_members[member] and written) else None

        scope_block = Block()
        scope_builder = Builder(InsertPoint.at_end(scope_block))

        # Per-dim index buffers for the kernel call, passed as MemRefType(i32, [ndim]).
        # Offset d_m[dim] un-normalizes the loop index back to the OPS global index:
        # ops_global_idx = normalized_loop_idx + d_m  (d_m is negative, e.g. -1)
        idx_buffers = []
        for _ in range(num_idx_args):
            idx_buffer = scope_builder.insert(memref.AllocaOp.get(i32, shape=[ndim]))
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
                idx_i32 = scope_builder.insert(arith.IndexCastOp(idx_op.idx, i32))
                dim_const = scope_builder.insert(
                    arith.ConstantOp(IntegerAttr(ndim - 1 - d, IndexType()))
                )
                scope_builder.insert(
                    memref.StoreOp.get(idx_i32.result, idx_buffer.memref, [dim_const.result])
                )
            idx_buffers.append(idx_buffer.memref)

        call_args = access_results + member_gbl + idx_buffers
        declare_kernel(
            module, kernel_name, [v.type for v in access_results] + [v.type for v in member_gbl],
            len(idx_buffers), ndim, num_results, result_elt
        )

        if num_results > 1:
            out_buf = scope_builder.insert(
                memref.AllocaOp.get(result_elt, shape=[num_results])
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
                func.CallOp(kernel_name, call_args, [result_elt] * num_results)
            )
            scope_results = list(kernel.results)

        scope_builder.insert(memref.AllocaScopeReturnOp.build(operands=[scope_results]))

        alloca_scope = memref.AllocaScopeOp.build(
            regions=[Region([scope_block])],
            result_types=[[result_elt] * num_results],
        )
        if cond is None:
            block_builder.insert(alloca_scope)
            results = list(alloca_scope.res)
        else:
            # Outside the member's range: yield the dats' previous values.
            # The alloca_scope sits inside the scf.if so the kernel (and its
            # index/out buffers) only runs where it is needed. Backends must
            # lower scf.if to control flow before memref.alloca_scope, which
            # cannot hold a multi-block region (see BackendPipeline.h).
            zero = tuple(0 for _ in range(ndim))
            olds = [
                current[a.dat.index.data] if a.dat.index.data in current
                else access(a.dat.index.data, zero)
                for a in written
            ]
            then_block = Block()
            then_builder = Builder(InsertPoint.at_end(then_block))
            then_builder.insert(alloca_scope)
            then_builder.insert(scf.YieldOp(*alloca_scope.res))
            else_block = Block()
            Builder(InsertPoint.at_end(else_block)).insert(scf.YieldOp(*olds))
            guarded_op = block_builder.insert(
                scf.IfOp(cond, [result_elt] * num_results, Region([then_block]), Region([else_block]))
            )
            results = list(guarded_op.results)
        last_results = results
        for a, value in zip(written, results):
            current[a.dat.index.data] = value

    if write_keys:
        returned = [current[k] for k in write_keys]
    else:
        # No dat outputs (e.g. a loop that only feeds a reduction): return the
        # kernel's own result, as the per-loop lowering always did.
        returned = list(last_results)
    block_builder.insert(stencil.ReturnOp.get(returned))

    fn_builder.insert(
        stencil.ApplyOp.build(
            operands=[
                [temp_of_field for temp_of_field in (field_of[k] for k in read_keys)]
                + gbl_block_args,
                [field_of[k] for k in write_keys],
                [],  # reduction operands empty for now
            ],
            regions=[Region([apply_block])],
            result_types=[[]],  # buffer semantic does not return results
            properties={"bounds": apply_bounds},
        )
    )

    fn_builder.insert(func.ReturnOp())
    return fn


@dataclass(frozen=True)
class OPSToStencilPass(ModulePass):
    """Lower ops.par_loop -> func.func + stencil.* (see module docstring).

    Loops sharing a `fuse_group` attribute become one function (see
    convert_group); loops without it are their own group.
    """
    name = "ops-to-stencil"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        loops = [o for o in op.body.block.ops if isinstance(o, ParLoopOp)]

        groups: dict[int, list[tuple[int, ParLoopOp]]] = {}
        for index, loop_op in enumerate(loops):
            gid = (
                loop_op.fuse_group.value.data
                if loop_op.fuse_group is not None
                else -(index + 1)
            )
            groups.setdefault(gid, []).append((index, loop_op))

        for gid, members in groups.items():
            indices = [i for i, _ in members]
            ops_in_group = [o for _, o in members]
            name = group_func_name(ops_in_group, indices, gid)
            op.body.block.add_op(convert_group(ops_in_group, name, op))

        for loop_op in loops:
            loop_op.detach()
            loop_op.erase()
