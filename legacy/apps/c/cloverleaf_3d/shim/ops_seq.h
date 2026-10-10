// Shim for the original CloverLeaf sources: they `#include "ops_seq.h"`, which in a
// stock build defines the sequential ops_par_loop. Here that include lands on this file
// (it is first on the include path) and
//   1. pulls in the real header with ops_par_loop renamed to ops_par_loop_stock, then
//   2. includes the ops-mlir wrapper, whose ops_par_loop queues loops for the JIT and
//      keeps the stock implementation as a fallback for loops it cannot compile.
// The CloverLeaf sources themselves are used unmodified, straight from the OPS tree.
#ifndef OPS_MLIR_CLOVERLEAF3D_SHIM_OPS_SEQ_H
#define OPS_MLIR_CLOVERLEAF3D_SHIM_OPS_SEQ_H

#define ops_par_loop ops_par_loop_stock
#include_next <ops_seq.h>
#undef ops_par_loop

#define OPS_MLIR_STOCK_FALLBACK 1
#include "ops/OPSWrapper.h"

// Globals that kernels read are declared with ops_decl_const (stock OPS needs that for
// its GPU backends). Registering them as well lets the JIT translate those reads; it
// snapshots their value every time a loop is enqueued, since e.g. dt changes per step.
namespace ops_mlir {
template <class T>
inline void declareConst(char const *name, int dim, char const *type, T *data) {
  ::ops_decl_const(name, dim, type, data);
  ops_register_kernel_constant(name, data);
}
} // namespace ops_mlir
#define ops_decl_const(name, dim, type, data) ops_mlir::declareConst(name, dim, type, data)

#endif
