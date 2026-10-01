//===- OPSWrapper.h - OPS JIT capture wrapper ----------------------*- C++
//-*-===//
//
// Part of OPS-MLIR Project
// Author: Prakanth Thilakaraj
// Date: June 2026
//
// This file is distributed under the MIT License.
// See LICENSE.txt for details.
//
// Intercepts OPS API calls to capture program structure for JIT compilation.
// Forwards to the real OPS library for correctness during development.
//
// Users only need: #include "ops/OPSWrapper.h"
//
//===----------------------------------------------------------------------===//

#ifndef OPS_WRAPPER_H
#define OPS_WRAPPER_H

#include "runtime/JITEngine.h"
#include "ops_lib_core.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

//===----------------------------------------------------------------------===//
// ops_par_loop interception
//===----------------------------------------------------------------------===//
// This template overrides the ops_par_loop to capture loop metadata.
// The captured loops are queued for JIT compilation.
//===----------------------------------------------------------------------===//

#ifdef OPS_MLIR_STOCK_FALLBACK
// Built via the shim ops_seq_v2.h (apps/c/cloverleaf_*/shim), which has renamed the
// stock OPS implementation to ops_par_loop_stock. Each queued loop carries a closure
// that runs it with that implementation, for loops the JIT cannot compile.
namespace ops_mlir {

template <typename KernelFn, std::size_t N, std::size_t... I>
void callStock(KernelFn kernel, const char *name, ops_block block, int dims,
               int *range, std::array<ops_arg, N> &args, std::index_sequence<I...>) {
  ops_par_loop_stock(kernel, name, block, dims, range, args[I]...);
}

template <typename KernelFn, typename... Args>
std::function<void()> makeStockFallback(KernelFn kernel, const char *name,
                                        ops_block block, int dims, const int *range,
                                        const Args &...opsArgs) {
  // The loop runs later, so read-only globals are copied now: eager execution
  // would have seen their current values.
  auto snapshots = std::make_shared<std::vector<std::vector<char>>>();
  std::array<ops_arg, sizeof...(Args)> args{opsArgs...};
  snapshots->reserve(args.size());
  for (ops_arg &a : args) {
    if (a.argtype == OPS_ARG_GBL && a.acc == OPS_READ && a.data) {
      snapshots->emplace_back(a.data, a.data + a.elem_size * a.dim);
      a.data = snapshots->back().data();
    }
  }
  std::array<int, 2 * OPS_MAX_DIM> r{};
  for (int i = 0; i < 2 * dims; ++i)
    r[i] = range[i];
  std::string kname = name;
  return [=]() mutable {
    callStock(kernel, kname.c_str(), block, dims, r.data(), args,
              std::make_index_sequence<sizeof...(Args)>{});
  };
}

} // namespace ops_mlir
#endif

template <typename KernelFn, typename... Args>
void ops_par_loop(KernelFn kernel, const char *name, ops_block block, int dims,
                  int *range, Args... opsArgs) {
  static_assert((std::is_same_v<std::decay_t<Args>, ops_arg> && ...),
                "All ops_par_loop variadic arguments must be ops_arg values");

  std::array<ops_arg, sizeof...(Args)> packedArgs{opsArgs...};

  auto token =
      static_cast<std::uintptr_t>(reinterpret_cast<std::uintptr_t>(kernel));

  std::function<void()> fallback;
#ifdef OPS_MLIR_STOCK_FALLBACK
  fallback = ops_mlir::makeStockFallback(kernel, name, block, dims, range,
                                         opsArgs...);
#endif

  ops_mlir::JITEngine::instance().enqueueParLoop(
      token, name, block, dims, range, packedArgs.data(), packedArgs.size(),
      std::move(fallback));
}

inline void compile_and_execute() {
  ops_mlir::JITEngine::instance().compile_and_execute();
}

// Equvalent to ops_decl_const
template <typename T>
void ops_register_kernel_constant(const char *name, T *data) {
  ops_mlir::JITEngine::instance().registerKernelConstant(name, data);
}

inline void set_kernel_source_file(const std::string &filePath) {
  ops_mlir::JITEngine::instance().setKernelSourceFile(filePath);
}

// Runs any queued loops and brings every device-resident dat back to the host.
// Call before reading `dat->data` directly; the OPS accessors below (fetch,
// raw pointer, ...) do this on their own.
inline void sync_all_host_buffers() {
  ops_mlir::JITEngine::instance().hostAccessAll();
}

namespace ops_mlir {
// Lazy execution: loops only run when something needs their results.
// These are the host-visible entry points that force that.
inline void hostRead(ops_dat dat) { JITEngine::instance().hostAccess(dat); }
inline void hostWrote(ops_dat dat) {
  JITEngine::instance().invalidateDeviceBuffer(
      reinterpret_cast<std::uintptr_t>(dat->data));
}
inline void hostReleased(ops_dat dat, ops_access acc) {
  if (acc != OPS_READ)
    hostWrote(dat);
}
inline void hostFlush() { JITEngine::instance().flushPending(); }

// With OPS_MLIR_STATS set, prints the runtime's cumulative counters on one
// line. Call it at fixed points of a time loop: the difference between two
// consecutive lines is an exact steady-state breakdown (no JIT noise).
inline void reportWindow(int iteration) {
  if (!std::getenv("OPS_MLIR_STATS"))
    return;
  const JITEngine::Stats &s = JITEngine::instance().stats();
  std::printf("[window] iter=%d loops=%zu launches=%zu compile=%.6f execute=%.6f "
              "kernel=%.6f halo=%.6f enqueue=%.6f plan=%.6f\n",
              iteration, s.numLoops, s.numLaunches, s.compileSeconds,
              s.executeSeconds, s.kernelSeconds, s.haloSeconds, s.enqueueSeconds,
              s.planSeconds);
  std::fflush(stdout);
}
} // namespace ops_mlir

// TODO: Use dat.data_d instead of deviceBuffers_. or improve the logic re copy only affected dats.
// Invalidate cached device buffers, forcing a host->device re-copy on next use.
#define ops_halo_transfer(group) ops_mlir::haloTransferIntercepted(group)

#define ops_NaNcheck(dat)                                                    \
  (ops_mlir::JITEngine::instance().syncHostBuffer(dat), ::ops_NaNcheck(dat))
#define ops_fetch_dat_hdf5_file(dat, file)                                   \
  (ops_mlir::JITEngine::instance().syncHostBuffer(dat),                      \
   ::ops_fetch_dat_hdf5_file(dat, file))

// Host-visible accessors: run the queued loops (and copy the dat back from the
// GPU) first; host writes invalidate the cached device copy.
#define ops_dat_get_raw_pointer(dat, part, stencil, memspace)                \
  (ops_mlir::hostRead(dat),                                                  \
   ::ops_dat_get_raw_pointer(dat, part, stencil, memspace))
#define ops_dat_release_raw_data(dat, part, acc)                             \
  (::ops_dat_release_raw_data(dat, part, acc),                               \
   ops_mlir::hostReleased(dat, acc))
#define ops_dat_fetch_data(dat, part, data)                                  \
  (ops_mlir::hostRead(dat), ::ops_dat_fetch_data(dat, part, data))
#define ops_dat_fetch_data_memspace(dat, part, data, memspace)               \
  (ops_mlir::hostRead(dat),                                                  \
   ::ops_dat_fetch_data_memspace(dat, part, data, memspace))
#define ops_dat_fetch_data_slab_memspace(dat, part, data, range, memspace)   \
  (ops_mlir::hostRead(dat),                                                  \
   ::ops_dat_fetch_data_slab_memspace(dat, part, data, range, memspace))
#define ops_dat_set_data(dat, part, data)                                    \
  (ops_mlir::hostRead(dat), ::ops_dat_set_data(dat, part, data),             \
   ops_mlir::hostWrote(dat))
#define ops_dat_set_data_memspace(dat, part, data, memspace)                 \
  (ops_mlir::hostRead(dat),                                                  \
   ::ops_dat_set_data_memspace(dat, part, data, memspace),                   \
   ops_mlir::hostWrote(dat))
#define ops_dat_set_data_slab_memspace(dat, part, data, range, memspace)     \
  (ops_mlir::hostRead(dat),                                                  \
   ::ops_dat_set_data_slab_memspace(dat, part, data, range, memspace),       \
   ops_mlir::hostWrote(dat))
#define ops_get_data(dat) (ops_mlir::hostRead(dat), ::ops_get_data(dat))
#define ops_print_dat_to_txtfile(dat, file)                                  \
  (ops_mlir::hostRead(dat), ::ops_print_dat_to_txtfile(dat, file))
#define ops_dat_copy(orig) (ops_mlir::hostRead(orig), ::ops_dat_copy(orig))
#define ops_dat_deep_copy(target, orig)                                      \
  (ops_mlir::hostRead(orig), ops_mlir::hostRead(target),                     \
   ::ops_dat_deep_copy(target, orig), ops_mlir::hostWrote(target))
#define ops_reduction_result(handle, ptr)                                    \
  (ops_mlir::hostFlush(), ::ops_reduction_result(handle, ptr))
#define ops_timing_output(stream)                                            \
  (ops_mlir::hostFlush(), ::ops_timing_output(stream))
#define ops_timing_output_stdout()                                           \
  (ops_mlir::hostFlush(), ::ops_timing_output_stdout())

// Tear down JITEngine's cached ExecutionEngines deterministically before
// the real ops_exit runs -- see JITEngine::shutdown's comment for why.
#define ops_exit() ops_mlir::exitIntercepted()

#endif // OPS_WRAPPER_H
