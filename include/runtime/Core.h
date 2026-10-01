
#ifndef OPS_MLIR_RUNTIME_CORE_H
#define OPS_MLIR_RUNTIME_CORE_H

#include "ops_lib_core.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ops_mlir {

enum class ArgKind { Dat, Gbl, Idx, Reduce, Unknown };

// Element type of a scalar/array global. OPS records only sizeof(T) for these, which
// cannot tell int from float, so the wrapper deduces it from the kernel signature.
// Unknown keeps the older "4 bytes is float, 8 bytes is double" reading.
enum ElemKind : int { EK_Unknown = 0, EK_F32 = 1, EK_F64 = 2, EK_I32 = 3, EK_I64 = 4 };

struct DatDesc {
  std::uintptr_t handle;
  int index;
  std::uintptr_t block;

  int dim;
  int type_size;
  int elem_size;

  std::vector<int64_t> size;
  std::vector<int64_t> base;
  std::vector<int64_t> d_m;
  std::vector<int64_t> d_p;
  std::vector<int64_t> stride;

  std::string name;
  std::string type;

  std::uintptr_t data;
  std::uintptr_t data_d;
};

struct StencilDesc {
  std::uintptr_t handle;
  int index;
  int dims;
  int points;
  std::string name;

  std::uintptr_t stencil;
  std::uintptr_t stride;
  std::uintptr_t mgrid_stride;
  int type;
};

struct ArgDesc {
  DatDesc dat;
  StencilDesc stencil;
  int dim;
  int elem_size;

  std::uintptr_t data;
  std::uintptr_t data_d;
  int acc;
  int argtype;
  int opt;
  int elem_kind = EK_Unknown; // ElemKind, set for globals
  // Added by the runtime, not passed by the application: the current value of a
  // registered constant the kernel reads (see KernelConstRef).
  bool synthetic = false;

  // Bytes of a read-only ops_arg_gbl, captured when the loop is enqueued so
  // that a host write between enqueue and flush cannot leak into the loop.
  std::vector<char> gbl_value;
};

struct LoopDesc {
  // Executes this loop with the stock OPS sequential implementation. Set by the
  // application wrapper when built with OPS_MLIR_STOCK_FALLBACK; loops the JIT
  // cannot compile (yet) run through it, in queue order. Empty otherwise.
  std::function<void()> fallback;

  std::string kernel_name;
  std::uintptr_t kernel_token;

  std::uintptr_t block;
  int dims;

  std::vector<int64_t> range;
  std::vector<ArgDesc> args;
};

}

#endif // OPS_MLIR_RUNTIME_CORE_H
