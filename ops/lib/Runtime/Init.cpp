//===- Init.cpp - libopsjit entry point --------------------------------------===//
//
// ops_jit_init(): called by the ops_init_backend() of every application translated with --jit, after ops_init and
// before the first ops_par_loop. For now it only decides whether the JIT is on; the executor (claim, lift, compile,
// launch) is added in Phase 3.
//
//===----------------------------------------------------------------------===//

#include "ops_jit.h"

#include <cstdlib>
#include <cstring>

namespace {
bool initialised = false;
bool enabled = true;
} // namespace

extern "C" void ops_jit_init(void) {
  if (initialised) return;
  initialised = true;
  const char *env = std::getenv("OPS_JIT");
  if (env && std::strcmp(env, "0") == 0) enabled = false;
  (void)enabled;
}
