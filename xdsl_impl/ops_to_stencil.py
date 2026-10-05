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
import math
from dataclasses import dataclass
from xdsl.builder import Builder, InsertPoint
from xdsl.context import Context
from xdsl.dialects import arith, func, memref, scf, stencil
from xdsl.dialects.builtin import FloatAttr, IndexType, IntegerAttr, MemRefType, ModuleOp, i32, i64, f32, f64
from xdsl.ir import Block, Region, SSAValue
from xdsl.passes import ModulePass

from ops_dialect import ArgType, Access, DatAttr, ParLoopOp, StencilAttr

def field_bounds(dat: DatAttr, axes: tuple[int, ...] | None = None) -> list[tuple[int, int]]:
    """Normalized 0-based field bounds: lb=0, ub=full allocated size per dim,
    reversed from OPS's [x, y, z] to put x last (unit-stride) -- see module note.
    `axes` keeps only those (reversed-order) axes: a dat indexed along fewer
    dimensions than the loop has a lower-rank field."""
    bounds = [(0, size) for size in reversed(dat.size_list)]
    return bounds if axes is None else [bounds[a] for a in axes]

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

def stencil_strides(stencil_attr: StencilAttr) -> list[int]:
    """Per-dim (OPS order) stride of an arg's stencil: 1 for a normal dim, 0 for a dim
    the dat does not have (an array indexed along fewer dimensions)."""
    dims = stencil_attr.dims.data
    addr = stencil_attr.stride.data
    if addr == 0:
        return [1] * dims
    return list((ctypes.c_int32 * dims).from_address(addr))


def kept_axes(stencil_attr: StencilAttr, ndim: int) -> tuple[int, ...]:
    """Axes (in the reversed order the lowering uses: x last) of the dat that a
    stencil actually indexes. Fewer than ndim for a strided stencil."""
    strides = stencil_strides(stencil_attr)
    return tuple(sorted(ndim - 1 - d for d in range(ndim) if strides[d] != 0))


def range_bounds(rng: list[int], ndim: int) -> list[tuple[int, int]]:
    return list(reversed([(rng[2 * i], rng[2 * i + 1]) for i in range(ndim)]))

def normalized_range_bounds(rng: list[int], d_m: list[int], ndim: int) -> list[tuple[int, int]]:
    return [(lb - dm, ub - dm) for (lb, ub), dm in zip(range_bounds(rng, ndim), d_m)]

_ELT_BY_TYPE_NAME = {"float": f32, "double": f64, "int": i32}
_ELT_BY_BYTES = {4: f32, 8: f64}
_ELT_BY_KIND = {1: f32, 2: f64, 3: i32, 4: i64}  # ElemKind in runtime/Core.h


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
    """MLIR element type of a global. OPS ignores the type string of
    ops_arg_gbl and only records sizeof(T); the wrapper deduces the element kind
    from the kernel signature when it can, and otherwise the size picks the type."""
    kind = arg.elem_kind.data
    if kind in _ELT_BY_KIND:
        return _ELT_BY_KIND[kind]
    nbytes = arg.elem_size.data
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


_REDUCTIONS = (Access.INC, Access.MIN, Access.MAX)


def reduction_identity(access: int, elt, builder) -> SSAValue:
    """The value that leaves a reduction unchanged: 0 for INC, +max for MIN, -max for MAX."""
    if elt in (f32, f64):
        v = {Access.INC: 0.0, Access.MIN: math.inf, Access.MAX: -math.inf}[access]
        return builder.insert(arith.ConstantOp(FloatAttr(v, elt))).result
    bits = elt.width.data
    v = {Access.INC: 0, Access.MIN: (1 << (bits - 1)) - 1, Access.MAX: -(1 << (bits - 1))}[access]
    return builder.insert(arith.ConstantOp(IntegerAttr(v, elt))).result


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
    dat_axes: dict[int, tuple[int, ...]] = {}  # axes the dat is indexed along
    for op in loops:
        for a in op.arg_list():
            if a.argtype.data == ArgType.DAT:
                key = a.dat.index.data
                axes = kept_axes(a.stencil, ndim)
                if key not in dat_attr:
                    dat_attr[key] = a.dat
                    dat_axes[key] = axes
                    dat_keys.append(key)
                elif dat_axes[key] != axes:
                    raise NotImplementedError(
                        f"dat '{a.dat.dat_name.data}' is accessed with stencils of "
                        "different strides in one group"
                    )
                if len(axes) < ndim and a.acc.data != Access.WRITE and a.acc.data != Access.READ:
                    raise NotImplementedError(
                        f"dat '{a.dat.dat_name.data}' is accumulated through a strided stencil"
                    )

    # A loop that writes dats indexed along only some of its dimensions iterates over those axes
    # alone (the runtime only compiles it when nothing depends on the dropped ones): the rest of
    # the loop would visit the same elements again.
    ndim_full = ndim
    written_axes = {dat_axes[a.dat.index.data] for op in loops for a in op.arg_list()
                    if a.argtype.data == ArgType.DAT and a.acc.data != Access.READ}
    act = tuple(range(ndim))
    if any(len(axes) < ndim for axes in written_axes):
        act = tuple(sorted({ax for axes in dat_axes.values() for ax in axes}))
        if written_axes != {act}:
            raise NotImplementedError(
                "a loop writing a dat indexed along fewer dimensions must have all its dats "
                "indexed along the same axes"
            )
        if any(a.argtype.data == ArgType.GBL and a.acc.data in _REDUCTIONS
               for op in loops for a in op.arg_list()):
            raise NotImplementedError("a loop over fewer axes cannot hold a reduction")
    act_pos = {ax: i for i, ax in enumerate(act)}
    dat_axes_full = dict(dat_axes)
    dat_axes = {k: tuple(act_pos[ax] for ax in axes) for k, axes in dat_axes_full.items()}
    ndim = len(act)

    def project(point):
        """Offsets of a stencil point along the active axes."""
        return tuple(point[ax] for ax in act)

    # d_m is shared by all dats on a block (see halo_offsets).
    # (a dat indexed along fewer dimensions has an unrelated d_m in the dimensions it lacks)
    # so each axis takes its d_m from a dat that has that axis.
    d_m_full = [None] * ndim_full
    for k in dat_keys:
        for axis in dat_axes_full[k]:
            if d_m_full[axis] is None:
                d_m_full[axis] = list(reversed(dat_attr[k].d_m_list))[axis]
    if any(d_m_full[ax] is None for ax in act):
        raise NotImplementedError("no dat of the loop spans one of its axes")
    d_m = [d_m_full[ax] for ax in act]

    # Bounding box of the members' (normalized) ranges.
    member_bounds = [
        [b for ax, b in enumerate(normalized_range_bounds(
            list(op.range.get_values()), [v or 0 for v in d_m_full], ndim_full)) if ax in act_pos]
        for op in loops
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
                # An array global is passed as one scalar per component.
                gbl_args.extend([a] * max(a.dim.data, 1))
    gbl_types = [gbl_elt(a) for a in gbl_args]

    # Reductions. A reduction argument is not accumulated in place: the kernel returns this
    # point's contribution (starting from the operation's identity) as one more result, and it is
    # stored in a scratch field over the group's box. The runtime allocates one scratch buffer per
    # element, passes them after the globals, and folds each into a single value on the device
    # afterwards (a separate small kernel) before combining it with the OPS reduction handle.
    red_elems = []   # (member, argument, element), in member / argument / element order
    for m, op in enumerate(loops):
        for a in op.arg_list():
            if a.argtype.data == ArgType.GBL and a.acc.data in _REDUCTIONS:
                red_elems.extend((m, a, k) for k in range(max(a.dim.data, 1)))
    red_types = [gbl_elt(a) for _, a, _ in red_elems]
    # Fields start at 0 like the dats' (the lowering takes the lower bound to be <= 0), so the part
    # below the box is never written: the runtime fills the buffer with the identity first.
    red_field_types = [stencil.FieldType([(0, hi) for _, hi in box], t) for t in red_types]

    field_types = {
        key: stencil.FieldType(field_bounds(dat_attr[key], dat_axes_full[key]), dat_elt(dat_attr[key]))
        for key in dat_keys
    }

    fn = func.FuncOp(
        fn_name,
        (tuple(field_types[k] for k in dat_keys) + tuple(gbl_types) + tuple(red_field_types), ()),
        visibility="private",
    )
    block = fn.body.block
    fn_builder = Builder(InsertPoint.at_end(block))
    field_of = dict(zip(dat_keys, block.args[: len(dat_keys)]))
    gbl_block_args = list(block.args[len(dat_keys): len(dat_keys) + len(gbl_types)])
    red_block_args = list(block.args[len(dat_keys) + len(gbl_types):])

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
            axes = dat_axes[key]
            if len(axes) == ndim:
                op = stencil.AccessOp(temp_of[key], point)
            else:
                # lower-rank dat: offsets along the axes it has, mapped onto the
                # loop's axes
                op = stencil.AccessOp(temp_of[key], [point[a] for a in axes], list(axes))
            accesses[memo] = block_builder.insert(op).res
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

    red_values: dict[int, SSAValue] = {}  # contribution of each reduction element
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
            for point in map(project, stencil_offsets(a.stencil)):
                if _is_zero(point) and key in current:
                    access_results.append(current[key])  # forwarded
                else:
                    access_results.append(access(key, point))

        n_gbl = sum(
            max(a.dim.data, 1) for a in args
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
        member_reds = [(i, e) for i, e in enumerate(red_elems) if e[0] == member]
        red_elt_set = {red_types[i] for i, _ in member_reds}
        if len(write_elts | red_elt_set) > 1:
            raise NotImplementedError(
                f"kernel '{kernel_name}' writes or reduces values of different element types; "
                "they must share a type."
            )
        result_elt = next(iter(write_elts | red_elt_set)) if (write_elts or red_elt_set) else f64
        num_results = (len(written) + len(member_reds)) or 1

        # Guarded member: run the kernel only where the point is inside the
        # member's range; elsewhere keep the dats' previous values.
        cond = (in_range(member_bounds[member])
                if (guarded_members[member] and (written or member_reds)) else None)

        scope_block = Block()
        scope_builder = Builder(InsertPoint.at_end(scope_block))

        # Per-dim index buffers for the kernel call, passed as MemRefType(i32, [ndim]).
        # Offset d_m[dim] un-normalizes the loop index back to the OPS global index:
        # ops_global_idx = normalized_loop_idx + d_m  (d_m is negative, e.g. -1)
        idx_buffers = []
        for _ in range(num_idx_args):
            idx_buffer = scope_builder.insert(memref.AllocaOp.get(i32, shape=[ndim_full]))
            for ax in range(ndim_full):
                if ax not in act_pos:  # a dropped axis: the kernel never reads it
                    zero_i32 = scope_builder.insert(arith.ConstantOp(IntegerAttr(0, i32)))
                    pos_const = scope_builder.insert(
                        arith.ConstantOp(IntegerAttr(ndim_full - 1 - ax, IndexType())))
                    scope_builder.insert(
                        memref.StoreOp.get(zero_i32.result, idx_buffer.memref, [pos_const.result]))
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
                    arith.ConstantOp(IntegerAttr(ndim_full - 1 - act[d], IndexType()))
                )
                scope_builder.insert(
                    memref.StoreOp.get(idx_i32.result, idx_buffer.memref, [dim_const.result])
                )
            idx_buffers.append(idx_buffer.memref)

        call_args = access_results + member_gbl + idx_buffers
        declare_kernel(
            module, kernel_name, [v.type for v in access_results] + [v.type for v in member_gbl],
            len(idx_buffers), ndim_full, num_results, result_elt
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
            else_builder = Builder(InsertPoint.at_end(else_block))
            # outside the member's range a reduction contributes nothing
            identities = [reduction_identity(red_elems[i][1].acc.data, red_types[i], else_builder)
                          for i, _ in member_reds]
            else_builder.insert(scf.YieldOp(*olds, *identities))
            guarded_op = block_builder.insert(
                scf.IfOp(cond, [result_elt] * num_results, Region([then_block]), Region([else_block]))
            )
            results = list(guarded_op.results)
        last_results = results
        for a, value in zip(written, results):
            current[a.dat.index.data] = value
        for (i, _), value in zip(member_reds, results[len(written):]):
            red_values[i] = value

    if write_keys or red_elems:
        returned = [current[k] for k in write_keys] + [red_values[i] for i in range(len(red_elems))]
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
                [field_of[k] for k in write_keys] + red_block_args,
                [],  # stencil.apply's own reduction operands are not used: see red_elems
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
