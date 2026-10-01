// Shim for the original CloverLeaf sources: they `#include "ops_seq_v2.h"`, which in a
// stock build defines the sequential ops_par_loop. Here that include lands on this file
// (it is first on the include path) and
//   1. pulls in the real header with ops_par_loop renamed to ops_par_loop_stock, then
//   2. includes the ops-mlir wrapper, whose ops_par_loop queues loops for the JIT and
//      keeps the stock implementation as a fallback for loops it cannot compile.
// The CloverLeaf sources themselves are used unmodified, straight from the OPS tree.
#ifndef OPS_MLIR_CLOVERLEAF_SHIM_OPS_SEQ_V2_H
#define OPS_MLIR_CLOVERLEAF_SHIM_OPS_SEQ_V2_H

#define ops_par_loop ops_par_loop_stock
#include_next <ops_seq_v2.h>
#undef ops_par_loop

#define OPS_MLIR_STOCK_FALLBACK 1
#include "ops/OPSWrapper.h"

#endif
