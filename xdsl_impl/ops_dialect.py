"""
Mirror of OPS_DatAttr, OPS_StencilAttr, and OPS_ArgAttr from OPS @include/Dialect/OPS/OPSOps.td

Part of OPS-MLIR Project
Author: Prakanth Thilakaraj
Date: June 2026

This file is distributed under the MIT License.
See LICENSE.txt for details.
"""

from enum import IntEnum

from xdsl.dialects.builtin import ArrayAttr, DenseArrayBase, IntAttr, IntegerAttr, StringAttr
from xdsl.ir import Dialect, ParametrizedAttribute
from xdsl.irdl import IRDLOperation, attr_def, irdl_attr_definition, irdl_op_definition, opt_attr_def
from xdsl.parser import AttrParser
from xdsl.printer import Printer


class Access(IntEnum):
    READ = 0
    WRITE = 1
    RW = 2
    INC = 3
    MIN = 4
    MAX = 5


class ArgType(IntEnum):
    GBL = 0
    DAT = 1
    IDX = 2


def _parse_int(parser: AttrParser) -> IntAttr:
    return IntAttr(parser.parse_integer())


def _print_int(printer: Printer, value: IntAttr) -> None:
    printer.print_string(str(value.data))


def _parse_int_list(parser: AttrParser) -> ArrayAttr[IntAttr]:
    """`[1, 2, 3]` -- stripped form of a DenseI64ArrayAttr nested parameter."""
    ints = parser.parse_comma_separated_list(
        parser.Delimiter.SQUARE, lambda: parser.parse_integer(allow_boolean=False)
    )
    return ArrayAttr(IntAttr(i) for i in ints)


def _print_int_list(printer: Printer, values: ArrayAttr[IntAttr]) -> None:
    printer.print_string(f"[{', '.join(str(v.data) for v in values)}]")


def _parse_str(parser: AttrParser) -> StringAttr:
    return StringAttr(parser.parse_str_literal())


def _print_str(printer: Printer, value: StringAttr) -> None:
    printer.print_string_literal(value.data)


@irdl_attr_definition
class DatAttr(ParametrizedAttribute):
    """Mirrors OPS_DatAttr / ops_dat_core."""

    name = "ops.dat"

    index: IntAttr
    block: IntAttr
    dim: IntAttr
    type_size: IntAttr
    elem_size: IntAttr
    size: ArrayAttr[IntAttr]
    base: ArrayAttr[IntAttr]
    d_m: ArrayAttr[IntAttr]
    d_p: ArrayAttr[IntAttr]
    stride: ArrayAttr[IntAttr]
    dat_name: StringAttr
    dat_type: StringAttr
    data: IntAttr
    data_d: IntAttr

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list:
        with parser.in_angle_brackets():
            index = _parse_int(parser)
            parser.parse_punctuation(",")
            block = _parse_int(parser)
            parser.parse_punctuation(",")
            dim = _parse_int(parser)
            parser.parse_punctuation(",")
            type_size = _parse_int(parser)
            parser.parse_punctuation(",")
            elem_size = _parse_int(parser)
            parser.parse_punctuation(",")
            size = _parse_int_list(parser)
            parser.parse_punctuation(",")
            base = _parse_int_list(parser)
            parser.parse_punctuation(",")
            d_m = _parse_int_list(parser)
            parser.parse_punctuation(",")
            d_p = _parse_int_list(parser)
            parser.parse_punctuation(",")
            stride = _parse_int_list(parser)
            parser.parse_punctuation(",")
            dat_name = _parse_str(parser)
            parser.parse_punctuation(",")
            dat_type = _parse_str(parser)
            parser.parse_punctuation(",")
            data = _parse_int(parser)
            parser.parse_punctuation(",")
            data_d = _parse_int(parser)
        return [index, block, dim, type_size, elem_size, size, base, d_m, d_p,
                stride, dat_name, dat_type, data, data_d]

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            _print_int(printer, self.index)
            printer.print_string(", ")
            _print_int(printer, self.block)
            printer.print_string(", ")
            _print_int(printer, self.dim)
            printer.print_string(", ")
            _print_int(printer, self.type_size)
            printer.print_string(", ")
            _print_int(printer, self.elem_size)
            printer.print_string(", ")
            _print_int_list(printer, self.size)
            printer.print_string(", ")
            _print_int_list(printer, self.base)
            printer.print_string(", ")
            _print_int_list(printer, self.d_m)
            printer.print_string(", ")
            _print_int_list(printer, self.d_p)
            printer.print_string(", ")
            _print_int_list(printer, self.stride)
            printer.print_string(", ")
            _print_str(printer, self.dat_name)
            printer.print_string(", ")
            _print_str(printer, self.dat_type)
            printer.print_string(", ")
            _print_int(printer, self.data)
            printer.print_string(", ")
            _print_int(printer, self.data_d)

    # Convenience accessors as plain python values.
    @property
    def size_list(self) -> list[int]:
        return [v.data for v in self.size]

    @property
    def d_m_list(self) -> list[int]:
        return [v.data for v in self.d_m]

    @property
    def d_p_list(self) -> list[int]:
        return [v.data for v in self.d_p]


@irdl_attr_definition
class StencilAttr(ParametrizedAttribute):
    """Mirrors OPS_StencilAttr / ops_stencil_core."""

    name = "ops.stencil"

    index: IntAttr
    dims: IntAttr
    points: IntAttr
    stencil_name: StringAttr
    stencil: IntAttr
    stride: IntAttr
    mgrid_stride: IntAttr
    type: IntAttr

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list:
        with parser.in_angle_brackets():
            index = _parse_int(parser)
            parser.parse_punctuation(",")
            dims = _parse_int(parser)
            parser.parse_punctuation(",")
            points = _parse_int(parser)
            parser.parse_punctuation(",")
            stencil_name = _parse_str(parser)
            parser.parse_punctuation(",")
            stencil = _parse_int(parser)
            parser.parse_punctuation(",")
            stride = _parse_int(parser)
            parser.parse_punctuation(",")
            mgrid_stride = _parse_int(parser)
            parser.parse_punctuation(",")
            type_ = _parse_int(parser)
        return [index, dims, points, stencil_name, stencil, stride, mgrid_stride, type_]

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            _print_int(printer, self.index)
            printer.print_string(", ")
            _print_int(printer, self.dims)
            printer.print_string(", ")
            _print_int(printer, self.points)
            printer.print_string(", ")
            _print_str(printer, self.stencil_name)
            printer.print_string(", ")
            _print_int(printer, self.stencil)
            printer.print_string(", ")
            _print_int(printer, self.stride)
            printer.print_string(", ")
            _print_int(printer, self.mgrid_stride)
            printer.print_string(", ")
            _print_int(printer, self.type)


@irdl_attr_definition
class ArgAttr(ParametrizedAttribute):
    """Mirrors OPS_ArgAttr / struct ops_arg."""

    name = "ops.arg"

    dat: DatAttr
    stencil: StencilAttr
    dim: IntAttr
    elem_size: IntAttr
    data: IntAttr
    data_d: IntAttr
    acc: IntAttr
    argtype: IntAttr
    opt: IntAttr

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list:
        with parser.in_angle_brackets():
            dat = DatAttr(*DatAttr.parse_parameters(parser))
            parser.parse_punctuation(",")
            stencil = StencilAttr(*StencilAttr.parse_parameters(parser))
            parser.parse_punctuation(",")
            dim = _parse_int(parser)
            parser.parse_punctuation(",")
            elem_size = _parse_int(parser)
            parser.parse_punctuation(",")
            data = _parse_int(parser)
            parser.parse_punctuation(",")
            data_d = _parse_int(parser)
            parser.parse_punctuation(",")
            acc = _parse_int(parser)
            parser.parse_punctuation(",")
            argtype = _parse_int(parser)
            parser.parse_punctuation(",")
            opt = _parse_int(parser)
        return [dat, stencil, dim, elem_size, data, data_d, acc, argtype, opt]

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            self.dat.print_parameters(printer)
            printer.print_string(", ")
            self.stencil.print_parameters(printer)
            printer.print_string(", ")
            _print_int(printer, self.dim)
            printer.print_string(", ")
            _print_int(printer, self.elem_size)
            printer.print_string(", ")
            _print_int(printer, self.data)
            printer.print_string(", ")
            _print_int(printer, self.data_d)
            printer.print_string(", ")
            _print_int(printer, self.acc)
            printer.print_string(", ")
            _print_int(printer, self.argtype)
            printer.print_string(", ")
            _print_int(printer, self.opt)


@irdl_op_definition
class ParLoopOp(IRDLOperation):
    name = "ops.par_loop"

    assembly_format = "attr-dict"

    kernel_ptr = attr_def(IntegerAttr)
    kernel_name = attr_def(StringAttr)
    dims = attr_def(IntegerAttr)
    range = attr_def(DenseArrayBase)
    args = attr_def(ArrayAttr)
    block_dims = opt_attr_def(IntegerAttr)
    fuse_group = opt_attr_def(IntegerAttr)

    def arg_list(self) -> list[ArgAttr]:
        return list(self.args.data)


OPS = Dialect("ops", [ParLoopOp], [DatAttr, StencilAttr, ArgAttr])
