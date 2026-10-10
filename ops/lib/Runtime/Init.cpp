//===- Init.cpp - libopsjit entry points ---------------------------------------===//
//
// ops_jit_init(): called by the ops_init_backend() of every application translated with --jit, after ops_init and
// before the first ops_par_loop. It reads the options and registers the chain executor with OPS (which then queues
// the loops).
//
//===----------------------------------------------------------------------===//

#include "ops_jit.h"

#include "ops-mlir/Runtime/Executor.h"
#include "ops-mlir/Runtime/Options.h"

#include "ops_lazy_executor.h"

#include <cstdio>
#include <cstdlib>

namespace {
bool initialised = false;

void atExit() {
  if (ops_mlir::runtime::options().stats) ops_mlir::runtime::executor().printStats();
}
} // namespace

extern "C" void ops_jit_init(void) {
  if (initialised) return;
  initialised = true;
  const ops_mlir::runtime::Options &opts = ops_mlir::runtime::options();
  if (!opts.enabled) return;
  ops_register_chain_executor(OPS_instance::getOPSInstance(), &ops_mlir::runtime::Executor::run, nullptr);
  std::atexit(atExit);
}

extern "C" int ops_jit_set_option(const char *name, const char *value) {
  return ops_mlir::runtime::options().set(name ? name : "", value ? value : "") ? 0 : 1;
}

extern "C" void ops_jit_print_stats(void) { ops_mlir::runtime::executor().printStats(); }
