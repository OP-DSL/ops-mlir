// End-to-end numerical tests for the ops-mlir JIT, built once per
// precision (OPS_TEST_F32 / OPS_TEST_F64). Every case enqueues OPS loops,
// flushes, and compares the result against a host reference computed in
// double precision. Backend comes from OPS_BACKEND (seq|openmp|cuda).

#include <cstdio>
#include <string>
#include <tuple>
#include <vector>

#define OPS_2D
#include "ops/OPSWrapper.h"

#include "../common/harness.h"

#if defined(OPS_TEST_F32)
#include "../kernels/kernels_f32.h"
#define TYPESTR "float"
#define PREC_NAME "f32"
#elif defined(OPS_TEST_F64)
#include "../kernels/kernels_f64.h"
#define TYPESTR "double"
#define PREC_NAME "f64"
#else
#error "define OPS_TEST_F32 or OPS_TEST_F64"
#endif

using namespace harness;
using ops_mlir::FusionOptions;

namespace {

constexpr int NX = 37; // deliberately non-square, non-power-of-two
constexpr int NY = 23;

struct Env {
  ops_block block;
  ops_stencil s00, s5;
  int full[4] = {-1, NX + 1, -1, NY + 1};
  int interior[4] = {0, NX, 0, NY};
  int bottom[4] = {-1, NX + 1, -1, 0};
  int top[4] = {-1, NX + 1, NY, NY + 1};
  int left[4] = {-1, 0, -1, NY + 1};
  int right[4] = {NX, NX + 1, -1, NY + 1};
};

ops_dat newDat(Env &e, const char *name) {
  int size[] = {NX, NY}, base[] = {0, 0}, d_m[] = {-1, -1}, d_p[] = {1, 1};
  return ops_decl_dat(e.block, 1, size, base, d_m, d_p, (real_t *)nullptr,
                      TYPESTR, name);
}

Field<real_t> view(ops_dat d) {
  return Field<real_t>{NX, NY, d->size[0], (const real_t *)d->data};
}

void flushAndSync() {
  compile_and_execute();
  sync_all_host_buffers();
}

// Launch accounting. beginCase() drains anything pending and zeroes the
// counters; expectLaunches() then asserts how many generated kernels the
// case needed: `guardedFused` with the default (different-range merging on),
// `sameRange` with only identical ranges merged, `unfused` = one per loop.
ops_mlir::JITEngine &jitEngine() { return ops_mlir::JITEngine::instance(); }

void beginCase() {
  jitEngine().flushPending();
  jitEngine().resetStats();
}

void expectLaunches(const char *name, size_t guardedFused, size_t sameRange,
                    size_t unfused) {
  const FusionOptions &o = jitEngine().fusionOptions();
  size_t want = !o.enabled ? unfused : o.allowGuarded ? guardedFused : sameRange;
  size_t got = jitEngine().stats().numLaunches;
  report(got == want, std::string(PREC_NAME) + "/" + name + "/launches",
         "launched " + std::to_string(got) + " kernels, expected " +
             std::to_string(want));
}

double initRef(int i, int j) { return i * 0.5 + j * 0.25 + 1.0; }

#define ARG_R(d, s) ops_arg_dat(d, 1, s, TYPESTR, OPS_READ)
#define ARG_W(d, s) ops_arg_dat(d, 1, s, TYPESTR, OPS_WRITE)
#define ARG_RW(d, s) ops_arg_dat(d, 1, s, TYPESTR, OPS_RW)

void check(const char *name, bool ok, const std::string &detail) {
  report(ok, std::string(PREC_NAME) + "/" + name, detail);
}

// a = 2b; c = a + 1; d = c * c (all point-wise on the interior).
void caseChain(Env &e) {
  ops_dat X = newDat(e, "cX"), Y = newDat(e, "cY"), Z = newDat(e, "cZ"),
          W = newDat(e, "cW");
  beginCase();
  ops_par_loop(k_init, "k_init", e.block, 2, e.full, ARG_W(X, e.s00), ops_arg_idx());
  ops_par_loop(k_scale2, "k_scale2", e.block, 2, e.interior, ARG_R(X, e.s00), ARG_W(Y, e.s00));
  ops_par_loop(k_plus1, "k_plus1", e.block, 2, e.interior, ARG_R(Y, e.s00), ARG_W(Z, e.s00));
  ops_par_loop(k_square, "k_square", e.block, 2, e.interior, ARG_R(Z, e.s00), ARG_W(W, e.s00));
  flushAndSync();
  std::string d;
  auto ref = [](int i, int j) { double c = 2 * initRef(i, j) + 1; return c * c; };
  expectLaunches("pointwise_chain", 1, 2, 4); // init(full) + interior chain: one guarded kernel
  check("pointwise_chain", compare<real_t>(view(W), ref, 0, NX, 0, NY, tolerance<real_t>(4), d), d);
}

// Non-local RAW: P = 2X on the full grid, Q = 5-point average of P.
void caseStencilRaw(Env &e) {
  ops_dat X = newDat(e, "rX"), P = newDat(e, "rP"), Q = newDat(e, "rQ");
  beginCase();
  ops_par_loop(k_init, "k_init", e.block, 2, e.full, ARG_W(X, e.s00), ops_arg_idx());
  ops_par_loop(k_scale2, "k_scale2", e.block, 2, e.full, ARG_R(X, e.s00), ARG_W(P, e.s00));
  ops_par_loop(k_lap5, "k_lap5", e.block, 2, e.interior, ARG_R(P, e.s5), ARG_W(Q, e.s00));
  flushAndSync();
  std::string d;
  auto ref = [](int i, int j) {
    return 0.25 * 2 * (initRef(i + 1, j) + initRef(i - 1, j) + initRef(i, j + 1) + initRef(i, j - 1));
  };
  expectLaunches("stencil_raw", 2, 2, 3); // init+scale2 fuse; the stencil read cannot join
  check("stencil_raw", compare<real_t>(view(Q), ref, 0, NX, 0, NY, tolerance<real_t>(4), d), d);
}

// Non-local RAW with IDENTICAL ranges, the case only the dependence check
// (not the range rules) keeps apart: P = 2X and Q = lap5(P) both over the
// interior. Q's boundary-adjacent points read P's halo, which is still zero.
// Fusing the two would feed Q the old P.
void caseStencilRawSameRange(Env &e) {
  ops_dat X = newDat(e, "sX2"), P = newDat(e, "sP2"), Q = newDat(e, "sQ2");
  ops_par_loop(k_init, "k_init", e.block, 2, e.full, ARG_W(X, e.s00), ops_arg_idx());
  ops_par_loop(k_zero, "k_zero", e.block, 2, e.full, ARG_W(P, e.s00));
  flushAndSync();
  beginCase();
  ops_par_loop(k_scale2, "k_scale2", e.block, 2, e.interior, ARG_R(X, e.s00), ARG_W(P, e.s00));
  ops_par_loop(k_lap5, "k_lap5", e.block, 2, e.interior, ARG_R(P, e.s5), ARG_W(Q, e.s00));
  flushAndSync();
  expectLaunches("stencil_raw_same_range", 2, 2, 2);
  auto p = [](int i, int j) {
    bool inside = i >= 0 && i < NX && j >= 0 && j < NY;
    return inside ? 2 * initRef(i, j) : 0.0;
  };
  auto ref = [&](int i, int j) { return 0.25 * (p(i + 1, j) + p(i - 1, j) + p(i, j + 1) + p(i, j - 1)); };
  std::string d;
  check("stencil_raw_same_range", compare<real_t>(view(Q), ref, 0, NX, 0, NY, tolerance<real_t>(3), d), d);
}

// Jacobi sweeps: Anew = lap5(A); A = Anew. Non-local WAR, must stay ordered.
void caseJacobi(Env &e) {
  const int iters = 20;
  ops_dat A = newDat(e, "jA"), B = newDat(e, "jB");
  ops_par_loop(k_init, "k_init", e.block, 2, e.full, ARG_W(A, e.s00), ops_arg_idx());
  ops_par_loop(k_zero, "k_zero", e.block, 2, e.full, ARG_W(B, e.s00));
  flushAndSync();
  beginCase();
  for (int it = 0; it < iters; ++it) {
    ops_par_loop(k_lap5, "k_lap5", e.block, 2, e.interior, ARG_R(A, e.s5), ARG_W(B, e.s00));
    ops_par_loop(k_copy, "k_copy", e.block, 2, e.interior, ARG_R(B, e.s00), ARG_W(A, e.s00));
    flushAndSync();
  }
  // Non-local WAR between lap5 (reads A 5pt) and copy (writes A): never fused.
  expectLaunches("jacobi_war", 2 * iters, 2 * iters, 2 * iters);
  // Host reference.
  const int sx = NX + 2;
  std::vector<double> a((NX + 2) * (NY + 2)), b(a.size(), 0.0);
  auto at = [&](std::vector<double> &v, int i, int j) -> double & { return v[(j + 1) * sx + (i + 1)]; };
  for (int j = -1; j <= NY; ++j)
    for (int i = -1; i <= NX; ++i) at(a, i, j) = initRef(i, j);
  for (int it = 0; it < iters; ++it) {
    for (int j = 0; j < NY; ++j)
      for (int i = 0; i < NX; ++i)
        at(b, i, j) = 0.25 * (at(a, i + 1, j) + at(a, i - 1, j) + at(a, i, j + 1) + at(a, i, j - 1));
    for (int j = 0; j < NY; ++j)
      for (int i = 0; i < NX; ++i) at(a, i, j) = at(b, i, j);
  }
  std::string d;
  auto ref = [&](int i, int j) { return at(a, i, j); };
  check("jacobi_war", compare<real_t>(view(A), ref, -1, NX + 1, -1, NY + 1, tolerance<real_t>(iters), d), d);
}

// Laplace-style setup: full-grid zero, then differently-ranged boundary
// writes to the same dat (WAW across ranges; corners must see the last write).
void caseBoundarySetup(Env &e) {
  ops_dat Y = newDat(e, "bY");
  beginCase();
  ops_par_loop(k_zero, "k_zero", e.block, 2, e.full, ARG_W(Y, e.s00));
  ops_par_loop(k_zero, "k_zero", e.block, 2, e.bottom, ARG_W(Y, e.s00));
  ops_par_loop(k_zero, "k_zero", e.block, 2, e.top, ARG_W(Y, e.s00));
  ops_par_loop(k_init, "k_init", e.block, 2, e.left, ARG_W(Y, e.s00), ops_arg_idx());
  ops_par_loop(k_init, "k_init", e.block, 2, e.right, ARG_W(Y, e.s00), ops_arg_idx());
  flushAndSync();
  std::string d;
  auto ref = [](int i, int j) { return (i == -1 || i == NX) ? initRef(i, j) : 0.0; };
  expectLaunches("boundary_setup", 1, 5, 5); // full grid + 4 strips in one guarded kernel
  check("boundary_setup", compare<real_t>(view(Y), ref, -1, NX + 1, -1, NY + 1, tolerance<real_t>(1), d), d);
}

// Multi-output kernel (out-struct) followed by a consumer of one output.
void caseMultiOutput(Env &e) {
  ops_dat X = newDat(e, "mX"), P = newDat(e, "mP"), Q = newDat(e, "mQ"), R = newDat(e, "mR");
  beginCase();
  ops_par_loop(k_init, "k_init", e.block, 2, e.full, ARG_W(X, e.s00), ops_arg_idx());
  ops_par_loop(k_two, "k_two", e.block, 2, e.interior, ARG_R(X, e.s00), ARG_W(P, e.s00), ARG_W(Q, e.s00));
  ops_par_loop(k_plus1, "k_plus1", e.block, 2, e.interior, ARG_R(P, e.s00), ARG_W(R, e.s00));
  flushAndSync();
  std::string d;
  bool ok = compare<real_t>(view(P), [](int i, int j) { return initRef(i, j) + 1; }, 0, NX, 0, NY, tolerance<real_t>(1), d) &&
            compare<real_t>(view(Q), [](int i, int j) { return initRef(i, j) * 2; }, 0, NX, 0, NY, tolerance<real_t>(1), d) &&
            compare<real_t>(view(R), [](int i, int j) { return initRef(i, j) + 2; }, 0, NX, 0, NY, tolerance<real_t>(2), d);
  expectLaunches("multi_output", 1, 2, 3);
  check("multi_output", ok, d);
}

// Read-write on one dat, chained, then a later write to a second dat.
void caseReadWrite(Env &e) {
  ops_dat X = newDat(e, "wX"), Y = newDat(e, "wY");
  beginCase();
  ops_par_loop(k_init, "k_init", e.block, 2, e.full, ARG_W(X, e.s00), ops_arg_idx());
  ops_par_loop(k_scale2, "k_scale2", e.block, 2, e.interior, ARG_RW(X, e.s00));
  ops_par_loop(k_plus1, "k_plus1", e.block, 2, e.interior, ARG_RW(X, e.s00));
  ops_par_loop(k_copy, "k_copy", e.block, 2, e.interior, ARG_R(X, e.s00), ARG_W(Y, e.s00));
  flushAndSync();
  std::string d;
  auto ref = [](int i, int j) { return 2 * initRef(i, j) + 1; };
  expectLaunches("read_write_chain", 1, 2, 4);
  check("read_write_chain",
        compare<real_t>(view(X), ref, 0, NX, 0, NY, tolerance<real_t>(2), d) &&
            compare<real_t>(view(Y), ref, 0, NX, 0, NY, tolerance<real_t>(2), d),
        d);
}

// Same loop shape flushed twice with different read-only global values:
// the second flush hits the compiled-module cache and must use the new value.
void caseGlobalCacheHit(Env &e) {
  ops_dat X = newDat(e, "gX"), Y = newDat(e, "gY");
  ops_par_loop(k_init, "k_init", e.block, 2, e.full, ARG_W(X, e.s00), ops_arg_idx());
  flushAndSync();
  real_t g = 2;
  ops_par_loop(k_gscale, "k_gscale", e.block, 2, e.interior, ARG_R(X, e.s00), ARG_W(Y, e.s00),
               ops_arg_gbl(&g, 1, TYPESTR, OPS_READ));
  flushAndSync();
  std::string d;
  bool ok = compare<real_t>(view(Y), [](int i, int j) { return 2 * initRef(i, j); }, 0, NX, 0, NY, tolerance<real_t>(1), d);
  g = 3;
  ops_par_loop(k_gscale, "k_gscale", e.block, 2, e.interior, ARG_R(X, e.s00), ARG_W(Y, e.s00),
               ops_arg_gbl(&g, 1, TYPESTR, OPS_READ));
  flushAndSync();
  ok = ok && compare<real_t>(view(Y), [](int i, int j) { return 3 * initRef(i, j); }, 0, NX, 0, NY, tolerance<real_t>(1), d);
  check("global_cache_hit", ok, d);
}

//===----------------------------------------------------------------------===//
// Lazy execution
//===----------------------------------------------------------------------===//

using ops_mlir::JITEngine;

// Loops must not run until something needs their result, and the OPS
// accessors must run them on demand (no explicit compile_and_execute()).
void caseLazyAutoFlush(Env &e) {
  auto &jit = JITEngine::instance();
  ops_dat X = newDat(e, "lX"), Y = newDat(e, "lY");
  jit.flushPending();
  jit.resetStats();
  ops_par_loop(k_init, "k_init", e.block, 2, e.full, ARG_W(X, e.s00), ops_arg_idx());
  ops_par_loop(k_scale2, "k_scale2", e.block, 2, e.interior, ARG_R(X, e.s00), ARG_W(Y, e.s00));
  bool deferred = jit.queue().size() == 2 && jit.stats().numFlushes == 0;
  check("lazy_defers_until_needed", deferred,
        "queue=" + std::to_string(jit.queue().size()) +
            " flushes=" + std::to_string(jit.stats().numFlushes));

  // ops_dat_fetch_data flushes and copies the interior (x-fastest, no halo).
  std::vector<real_t> buf((size_t)NX * NY, -1);
  ops_dat_fetch_data(Y, 0, (char *)buf.data());
  bool ok = jit.queue().empty() && jit.stats().numFlushes == 1;
  double worst = 0;
  for (int j = 0; j < NY; ++j)
    for (int i = 0; i < NX; ++i)
      worst = std::fmax(worst, std::fabs(buf[(size_t)j * NX + i] - 2 * initRef(i, j)));
  check("autoflush_fetch_data", ok && worst <= tolerance<real_t>(1) * 100,
        "flushes=" + std::to_string(jit.stats().numFlushes) +
            " worst=" + std::to_string(worst));

  // Direct dat->data access after get_raw_pointer sees flushed values.
  ops_par_loop(k_plus1, "k_plus1", e.block, 2, e.interior, ARG_R(Y, e.s00), ARG_W(X, e.s00));
  ops_memspace ms = OPS_HOST;
  ops_dat_get_raw_pointer(X, 0, e.s00, &ms);
  ops_dat_release_raw_data(X, 0, OPS_READ);
  std::string d;
  check("autoflush_raw_pointer",
        jit.queue().empty() &&
            compare<real_t>(view(X), [](int i, int j) { return 2 * initRef(i, j) + 1; }, 0, NX, 0, NY, tolerance<real_t>(2), d),
        d);
}

// Read-only globals are captured at enqueue time, not at flush time.
void caseGlobalSnapshot(Env &e) {
  ops_dat X = newDat(e, "sX"), Y = newDat(e, "sY"), Z = newDat(e, "sZ");
  beginCase();
  ops_par_loop(k_init, "k_init", e.block, 2, e.full, ARG_W(X, e.s00), ops_arg_idx());
  real_t g = 2;
  ops_par_loop(k_gscale, "k_gscale", e.block, 2, e.interior, ARG_R(X, e.s00), ARG_W(Y, e.s00),
               ops_arg_gbl(&g, 1, TYPESTR, OPS_READ));
  g = 3; // changes before the first loop has run
  ops_par_loop(k_gscale, "k_gscale", e.block, 2, e.interior, ARG_R(X, e.s00), ARG_W(Z, e.s00),
               ops_arg_gbl(&g, 1, TYPESTR, OPS_READ));
  g = 99;
  flushAndSync();
  std::string d;
  bool ok = compare<real_t>(view(Y), [](int i, int j) { return 2 * initRef(i, j); }, 0, NX, 0, NY, tolerance<real_t>(1), d) &&
            compare<real_t>(view(Z), [](int i, int j) { return 3 * initRef(i, j); }, 0, NX, 0, NY, tolerance<real_t>(1), d);
  expectLaunches("global_snapshot_at_enqueue", 1, 2, 3); // init + both gscale loops, each with its own g
  check("global_snapshot_at_enqueue", ok, d);
}

// A long queue is flushed automatically once it reaches the cap.
void caseQueueCap(Env &e) {
  auto &jit = JITEngine::instance();
  ops_dat X = newDat(e, "qX");
  jit.flushPending();
  jit.resetStats();
  jit.setQueueMax(3);
  ops_par_loop(k_init, "k_init", e.block, 2, e.full, ARG_W(X, e.s00), ops_arg_idx());
  for (int n = 0; n < 6; ++n)
    ops_par_loop(k_plus1, "k_plus1", e.block, 2, e.interior, ARG_RW(X, e.s00));
  bool capped = jit.stats().numFlushes == 2 && jit.queue().size() == 1;
  jit.setQueueMax(512);
  flushAndSync();
  std::string d;
  check("queue_cap_autoflush",
        capped && compare<real_t>(view(X), [](int i, int j) { return initRef(i, j) + 6; }, 0, NX, 0, NY, tolerance<real_t>(6), d),
        "flushes=" + std::to_string(jit.stats().numFlushes) + " " + d);
}

// Identical queues compile once.
void caseModuleCache(Env &e) {
  auto &jit = JITEngine::instance();
  ops_dat X = newDat(e, "mcX"), Y = newDat(e, "mcY");
  jit.flushPending();
  jit.resetStats();
  for (int rep = 0; rep < 3; ++rep) {
    // A loop shape no other case uses (init over the interior only, then
    // square), so the first flush is a genuine module-cache miss.
    ops_par_loop(k_init, "k_init", e.block, 2, e.interior, ARG_W(X, e.s00), ops_arg_idx());
    ops_par_loop(k_square, "k_square", e.block, 2, e.interior, ARG_R(X, e.s00), ARG_W(Y, e.s00));
    jit.flushPending();
  }
  check("module_cache_reuse", jit.stats().numCompiles == 1 && jit.stats().numFlushes == 3,
        "compiles=" + std::to_string(jit.stats().numCompiles) +
            " flushes=" + std::to_string(jit.stats().numFlushes));
}

//===----------------------------------------------------------------------===//
// Fused vs. unfused
//===----------------------------------------------------------------------===//

struct Outputs {
  ops_dat X, W, P, Q, R;
};

// A mixed workload: point-wise chain, multi-output kernel, RW updates, a
// global, a stencil consumer and a boundary strip. Run identically with
// fusion on and off; both must agree.
Outputs workload(Env &e, const char *tag) {
  auto name = [&](const char *n) { return std::string(tag) + n; };
  ops_dat X = newDat(e, name("X").c_str()), Y = newDat(e, name("Y").c_str()),
          Z = newDat(e, name("Z").c_str()), W = newDat(e, name("W").c_str()),
          P = newDat(e, name("P").c_str()), Q = newDat(e, name("Q").c_str()),
          R = newDat(e, name("R").c_str());
  real_t g = RLIT(1.5);
  ops_par_loop(k_init, "k_init", e.block, 2, e.full, ARG_W(X, e.s00), ops_arg_idx());
  ops_par_loop(k_scale2, "k_scale2", e.block, 2, e.interior, ARG_R(X, e.s00), ARG_W(Y, e.s00));
  ops_par_loop(k_plus1, "k_plus1", e.block, 2, e.interior, ARG_R(Y, e.s00), ARG_W(Z, e.s00));
  ops_par_loop(k_gscale, "k_gscale", e.block, 2, e.interior, ARG_R(Z, e.s00), ARG_W(W, e.s00),
               ops_arg_gbl(&g, 1, TYPESTR, OPS_READ));
  ops_par_loop(k_two, "k_two", e.block, 2, e.interior, ARG_R(W, e.s00), ARG_W(P, e.s00), ARG_W(Q, e.s00));
  ops_par_loop(k_add, "k_add", e.block, 2, e.interior, ARG_R(P, e.s00), ARG_R(Q, e.s00), ARG_W(R, e.s00));
  ops_par_loop(k_scale2, "k_scale2", e.block, 2, e.interior, ARG_RW(X, e.s00));
  ops_par_loop(k_lap5, "k_lap5", e.block, 2, e.interior, ARG_R(W, e.s5), ARG_W(Y, e.s00));
  flushAndSync();
  return {X, W, P, Q, R};
}

void caseFusedVsUnfused(Env &e) {
  auto &jit = jitEngine();
  FusionOptions saved = jit.fusionOptions();

  FusionOptions on = saved, off = saved;
  on.enabled = true;
  off.enabled = false;

  jit.setFusionOptions(off);
  beginCase();
  Outputs u = workload(e, "u");
  size_t unfusedLaunches = jit.stats().numLaunches;

  jit.setFusionOptions(on);
  beginCase();
  Outputs f = workload(e, "f");
  size_t fusedLaunches = jit.stats().numLaunches;
  jit.setFusionOptions(saved);

  // Bitwise on the CPU backends; the GPU may contract a*b+c differently once
  // more arithmetic shares a kernel, so allow a few ulp there.
  bool cpu = jit.backend() != ops_mlir::Backend::CUDA;
  std::string d;
  bool same = true;
  for (auto [a, b, n] : {std::tuple{u.X, f.X, "X"}, {u.W, f.W, "W"}, {u.P, f.P, "P"},
                         {u.Q, f.Q, "Q"}, {u.R, f.R, "R"}}) {
    if (cpu) {
      if (!identical<real_t>(view(a), view(b), 0, NX, 0, NY)) {
        same = false;
        d += std::string("differs: ") + n + " ";
      }
    } else {
      Field<real_t> fa = view(a);
      same = same && compare<real_t>(view(b), [&](int i, int j) { return fa.at(i, j); },
                                     0, NX, 0, NY, tolerance<real_t>(4), d);
    }
  }
  check(cpu ? "fused_equals_unfused_bitwise" : "fused_equals_unfused_ulp", same, d);
  report(jit.fusionOptions().enabled ? fusedLaunches < unfusedLaunches : true,
         std::string(PREC_NAME) + "/fusion_reduces_launches",
         "fused=" + std::to_string(fusedLaunches) + " unfused=" + std::to_string(unfusedLaunches));
}

} // namespace

int main(int argc, const char **argv) {
  set_kernel_source_file(OPS_TEST_KERNEL_HEADER);
  ops_init(argc, argv, 1);

  Env e;
  e.block = ops_decl_block(2, "test_grid");
  int s00[] = {0, 0};
  e.s00 = ops_decl_stencil(2, 1, s00, "0,0");
  int s5[] = {0, 0, 1, 0, -1, 0, 0, 1, 0, -1};
  e.s5 = ops_decl_stencil(2, 5, s5, "5pt");
  ops_partition("");

  std::printf("precision=%s\n", PREC_NAME);
  caseChain(e);
  caseStencilRaw(e);
  caseStencilRawSameRange(e);
  caseJacobi(e);
  caseBoundarySetup(e);
  caseMultiOutput(e);
  caseReadWrite(e);
  caseGlobalCacheHit(e);
  caseLazyAutoFlush(e);
  caseGlobalSnapshot(e);
  caseQueueCap(e);
  caseModuleCache(e);
  caseFusedVsUnfused(e);

  std::printf("%d passed, %d failed\n", passes, failures);
  ops_exit();
  return failures ? 1 : 0;
}
