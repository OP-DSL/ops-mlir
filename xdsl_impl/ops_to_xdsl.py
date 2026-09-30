"""
Lower ops.par_loop -> func.func + stencil.* using xDSL, mirroring
lib/passes/OPSToStencil.cpp's C++ design (see xdsl_impl/laplace_stencil_1.mlir
for the target IR shape).

Usage:
    python xdsl_impl/ops_to_xdsl.py xdsl_impl/laplace_sample.mlir
"""

import sys
from pathlib import Path

from xdsl.context import Context
from xdsl.dialects import func, stencil
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.transforms.experimental.convert_stencil_to_ll_mlir import (
    ConvertStencilToLLMLIRPass,
)
from xdsl.passes import PassPipeline


sys.path.insert(0, str(Path(__file__).parent))
from ops_dialect import OPS
from ops_to_stencil import OPSToStencilPass

def convert_ir_text(text: str) -> str:
    """String-in/string-out entry point for embedding (see
    lib/runtime/JITEngine.cpp::runXdslLowering). Unlike main(), this never
    touches the filesystem or stdout/stderr -- the caller owns the IR
    string and the result string.
    """
    import os
    if os.environ.get("OPS_MLIR_XDSL_PROFILE"):
        return _profiled(text)
    return _convert_ir_text(text)


_profiler = None


def _profiled(text: str) -> str:
    """OPS_MLIR_XDSL_PROFILE=1: accumulate a cProfile over every lowering and print
    the top entries (by cumulative time) when the process exits."""
    global _profiler
    import atexit, cProfile, pstats, sys
    if _profiler is None:
        _profiler = cProfile.Profile()
        atexit.register(lambda: pstats.Stats(_profiler, stream=sys.stderr)
                        .sort_stats("cumulative").print_stats(30))
    _profiler.enable()
    try:
        return _convert_ir_text(text)
    finally:
        _profiler.disable()


def _memoize_type_conversion() -> None:
    """xDSL's TypeConversionPattern re-converts the type of every result, attribute
    and block argument of every operation, recursing through each attribute with
    several isinstance/constraint checks. A module has a few thousand operations but
    only a handful of distinct types, and that conversion was ~90% of the lowering
    time (minutes for a CloverLeaf time step). Attributes are immutable and hashable
    and the conversions are pure, so cache them per pattern instance."""
    from xdsl.pattern_rewriter import TypeConversionPattern
    if getattr(TypeConversionPattern, "_ops_mlir_memoized", False):
        return
    original = TypeConversionPattern._convert_type_rec
    cache: dict = {}
    keep_alive: list = []  # so an id() in the cache is never reused

    def memoized(self, typ):
        key = (id(self), typ)
        try:
            return cache[key]
        except KeyError:
            pass
        except TypeError:  # unhashable attribute: just convert it
            return original(self, typ)
        if not any(p is self for p in keep_alive[-4:]):
            keep_alive.append(self)
        result = original(self, typ)
        cache[key] = result
        return result

    TypeConversionPattern._convert_type_rec = memoized
    TypeConversionPattern._ops_mlir_memoized = True


_memoize_type_conversion()


def _convert_ir_text(text: str) -> str:
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(OPS)
    ctx.load_dialect(stencil.Stencil)
    ctx.load_dialect(func.Func)

    module = Parser(ctx, text).parse_module()
    import os
    if os.environ.get("OPS_MLIR_DUMP_STENCIL"):
        # the IR between the two passes: stencil.apply / stencil.access, before it becomes loops
        PassPipeline([OPSToStencilPass()]).apply(ctx, module)
        import sys
        print("=== STENCIL IR (after ops-to-stencil) ===", file=sys.stderr)
        Printer(stream=sys.stderr).print_op(module)
        print(file=sys.stderr)
        PassPipeline([ConvertStencilToLLMLIRPass()]).apply(ctx, module)
    else:
        pipeline = PassPipeline([
            OPSToStencilPass(),
            ConvertStencilToLLMLIRPass(),
        ])
        pipeline.apply(ctx, module)
    module.verify()

    from io import StringIO

    out = StringIO()
    Printer(stream=out).print_op(module)
    return out.getvalue()


def main() -> None:
    args = sys.argv[1:]

    if len(args) != 1:
        print(f"usage: {sys.argv[0]} <input.mlir>", file=sys.stderr)
        sys.exit(1)

    text = Path(args[0]).read_text()
    print(convert_ir_text(text))


if __name__ == "__main__":
    main()
