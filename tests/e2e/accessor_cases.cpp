// End-to-end tests for the accessor-style kernel path (what CloverLeaf uses): reductions,
// loops that write dats indexed along fewer axes, and registered arrays of structs. Each case
// checks the result against a host reference and that no loop fell back to the stock OPS
// implementation. Backend comes from OPS_BACKEND (seq|openmp|cuda); everything is double.

#include <cmath>
#include <cstdio>
#include <string>

#define OPS_2D
#include "ops_seq_v2.h" // the shim: stock OPS under another name, then the ops-mlir wrapper

#include "../common/harness.h"

#include "../kernels/accessor_kernels.h"

tstate_type *tstates = nullptr;
int tnum_states = 0;

using namespace harness;

namespace {

constexpr int NX = 37;
constexpr int NY = 23;

ops_mlir::JITEngine &engine() { return ops_mlir::JITEngine::instance(); }

double initRef(int i, int j) { return i * 0.5 + j * 0.25 + 1.0; }

struct Env {
  ops_block block;
  ops_stencil s00, sx1, s00x, s00y, s00x1;
  int full[4] = {-1, NX + 1, -1, NY + 1};
  int interior[4] = {0, NX, 0, NY};
};

ops_dat newDat(Env &, ops_block block, const char *name) {
  int size[] = {NX, NY}, base[] = {0, 0}, d_m[] = {-1, -1}, d_p[] = {1, 1};
  return ops_decl_dat(block, 1, size, base, d_m, d_p, (double *)nullptr, "double", name);
}

// a dat indexed along one axis (the other has extent 1)
ops_dat newLine(Env &, ops_block block, const char *name, bool alongX) {
  int size[] = {alongX ? NX : 1, alongX ? 1 : NY}, base[] = {0, 0};
  int d_m[] = {alongX ? -1 : 0, alongX ? 0 : -1}, d_p[] = {alongX ? 2 : 0, alongX ? 0 : 2};
  return ops_decl_dat(block, 1, size, base, d_m, d_p, (double *)nullptr, "double", name);
}

void begin() {
  engine().flushPending();
  engine().resetStats();
}

void finish(const char *name, bool ok, const std::string &detail) {
  std::size_t host = engine().stats().numHostLoops;
  report(ok && host == 0, std::string("accessor/") + name,
         ok ? std::to_string(host) + " loops ran through the stock fallback" : detail);
}

bool near(double got, double want, double rel) {
  return std::fabs(got - want) <= rel * (1.0 + std::fabs(want));
}

void caseReductions(Env &e) {
  ops_dat X = newDat(e, e.block, "rX");
  ops_reduction sum = ops_decl_reduction_handle(sizeof(double), "double", "sum");
  ops_reduction lo = ops_decl_reduction_handle(sizeof(double), "double", "lo");
  ops_reduction hi = ops_decl_reduction_handle(sizeof(double), "double", "hi");
  ops_reduction sum2 = ops_decl_reduction_handle(sizeof(double), "double", "sum2");
  ops_reduction sq = ops_decl_reduction_handle(sizeof(double), "double", "sq");
  ops_reduction two = ops_decl_reduction_handle(2 * sizeof(double), "double", "two");
  begin();
  ops_par_loop(a_init, "a_init", e.block, 2, e.full,
               ops_arg_dat(X, 1, e.s00, "double", OPS_WRITE), ops_arg_idx());
  ops_par_loop(a_stats, "a_stats", e.block, 2, e.interior,
               ops_arg_dat(X, 1, e.s00, "double", OPS_READ),
               ops_arg_reduce(sum, 1, "double", OPS_INC), ops_arg_reduce(lo, 1, "double", OPS_MIN),
               ops_arg_reduce(hi, 1, "double", OPS_MAX));
  ops_par_loop(a_sum_spelled, "a_sum_spelled", e.block, 2, e.interior,
               ops_arg_dat(X, 1, e.s00, "double", OPS_READ),
               ops_arg_reduce(sum2, 1, "double", OPS_INC), ops_arg_reduce(sq, 1, "double", OPS_INC));
  ops_par_loop(a_sum2, "a_sum2", e.block, 2, e.interior, ops_arg_dat(X, 1, e.s00, "double", OPS_READ),
               ops_arg_reduce(two, 2, "double", OPS_INC));
  double s, l, h, s2, q, t[2];
  ops_reduction_result(sum, &s);
  ops_reduction_result(lo, &l);
  ops_reduction_result(hi, &h);
  ops_reduction_result(sum2, &s2);
  ops_reduction_result(sq, &q);
  ops_reduction_result(two, t);
  double rs = 0, rl = 1e300, rh = -1e300, rq = 0;
  for (int j = 0; j < NY; ++j)
    for (int i = 0; i < NX; ++i) {
      double v = initRef(i, j);
      rs += v;
      rl = std::min(rl, v);
      rh = std::max(rh, v);
      rq += v * v;
    }
  char detail[256];
  std::snprintf(detail, sizeof detail, "sum %.17g/%.17g min %.17g/%.17g max %.17g/%.17g sq %.17g/%.17g",
                s, rs, l, rl, h, rh, q, rq);
  finish("reductions",
         near(s, rs, 1e-13) && l == rl && h == rh && near(s2, rs, 1e-13) && near(q, rq, 1e-13) &&
             near(t[0], rs, 1e-13) && near(t[1], 2 * rs, 1e-13),
         detail);

  // a second round: the handles accumulate across loops until their result is read
  begin();
  ops_par_loop(a_stats, "a_stats", e.block, 2, e.interior,
               ops_arg_dat(X, 1, e.s00, "double", OPS_READ),
               ops_arg_reduce(sum, 1, "double", OPS_INC), ops_arg_reduce(lo, 1, "double", OPS_MIN),
               ops_arg_reduce(hi, 1, "double", OPS_MAX));
  ops_par_loop(a_stats, "a_stats", e.block, 2, e.interior,
               ops_arg_dat(X, 1, e.s00, "double", OPS_READ),
               ops_arg_reduce(sum, 1, "double", OPS_INC), ops_arg_reduce(lo, 1, "double", OPS_MIN),
               ops_arg_reduce(hi, 1, "double", OPS_MAX));
  ops_reduction_result(sum, &s);
  ops_reduction_result(lo, &l);
  ops_reduction_result(hi, &h);
  finish("reductions_twice", near(s, 2 * rs, 1e-13) && l == rl && h == rh, "two loops into one handle");

  // a loop over a single point may assign the reduction
  ops_reduction pick = ops_decl_reduction_handle(sizeof(double), "double", "pick");
  int one[4] = {5, 6, 3, 4};
  begin();
  ops_par_loop(a_pick, "a_pick", e.block, 2, one, ops_arg_dat(X, 1, e.s00, "double", OPS_READ),
               ops_arg_reduce(pick, 1, "double", OPS_INC));
  double p;
  ops_reduction_result(pick, &p);
  finish("reduction_single_point", p == initRef(5, 3), "value " + std::to_string(p));
}

void caseLines(Env &e) {
  ops_dat CX = newLine(e, e.block, "CX", true), CY = newLine(e, e.block, "CY", false);
  ops_dat CELL = newLine(e, e.block, "CELL", true), O = newDat(e, e.block, "O");
  int allx[4] = {-1, NX + 2, -1, 0}, ally[4] = {-1, 0, -1, NY + 2};
  int cellx[4] = {-1, NX + 1, -1, 0};
  begin();
  // over the whole 2-D box: the loop visits each element of the 1-D arrays once per other index
  int box[4] = {-1, NX + 2, -1, NY + 2};
  ops_par_loop(a_coord_x, "a_coord_x", e.block, 2, box,
               ops_arg_dat(CX, 1, e.s00x, "double", OPS_WRITE), ops_arg_idx());
  ops_par_loop(a_coord_y, "a_coord_y", e.block, 2, ally,
               ops_arg_dat(CY, 1, e.s00y, "double", OPS_WRITE), ops_arg_idx());
  ops_par_loop(a_coord_cell, "a_coord_cell", e.block, 2, cellx,
               ops_arg_dat(CX, 1, e.s00x1, "double", OPS_READ),
               ops_arg_dat(CELL, 1, e.s00x, "double", OPS_WRITE));
  ops_par_loop(a_outer, "a_outer", e.block, 2, e.interior,
               ops_arg_dat(CX, 1, e.s00x, "double", OPS_READ),
               ops_arg_dat(CY, 1, e.s00y, "double", OPS_READ),
               ops_arg_dat(O, 1, e.s00, "double", OPS_WRITE));
  (void)allx;
  compile_and_execute();
  sync_all_host_buffers();
  const double *o = (const double *)O->data;
  const double *cell = (const double *)CELL->data;
  bool ok = true;
  char detail[200] = "";
  for (int j = 0; j < NY && ok; ++j)
    for (int i = 0; i < NX && ok; ++i) {
      double want = (i * 1.5 - 2.0) * (j * 0.75 + 4.0), got = o[(j + 1) * O->size[0] + (i + 1)];
      if (!near(got, want, 1e-14)) {
        ok = false;
        std::snprintf(detail, sizeof detail, "O(%d,%d) = %.17g, expected %.17g", i, j, got, want);
      }
    }
  for (int i = -1; i < NX && ok; ++i) {
    double want = 0.5 * ((i * 1.5 - 2.0) + ((i + 1) * 1.5 - 2.0)), got = cell[i + 1];
    if (!near(got, want, 1e-14)) {
      ok = false;
      std::snprintf(detail, sizeof detail, "CELL(%d) = %.17g, expected %.17g", i, got, want);
    }
  }
  finish("lines_written_over_fewer_axes", ok, detail);
}

void caseStates(Env &e) {
  static tstate_type table[3] = {{2.0, 1.0, 1}, {5.0, 7.0, 0}, {0.5, -1.0, 1}};
  tstates = table;
  tnum_states = 3;
  ops_decl_const("tstates", 3, "tstate_type", tstates);
  ops_decl_const("tnum_states", 1, "int", &tnum_states);
  ops_dat Y = newDat(e, e.block, "sY");
  begin();
  ops_par_loop(a_states, "a_states", e.block, 2, e.interior, ops_arg_dat(Y, 1, e.s00, "double", OPS_WRITE));
  compile_and_execute();
  sync_all_host_buffers();
  double want = 1.0;
  for (int s = 0; s < tnum_states; ++s)
    if (table[s].kind == 1)
      want = want * table[s].scale + table[s].shift;
  const double got = ((const double *)Y->data)[(3 + 1) * Y->size[0] + 4 + 1];
  finish("array_of_structs_constant", got == want, std::to_string(got) + " vs " + std::to_string(want));

  // the code is specialised on the trip count, so a different count must not reuse the module
  begin();
  tnum_states = 2;
  ops_par_loop(a_states, "a_states", e.block, 2, e.interior, ops_arg_dat(Y, 1, e.s00, "double", OPS_WRITE));
  compile_and_execute();
  sync_all_host_buffers();
  want = 1.0 * table[0].scale + table[0].shift; // states[1].kind == 0
  const double got2 = ((const double *)Y->data)[(3 + 1) * Y->size[0] + 4 + 1];
  finish("specialised_constant_in_module_key", got2 == want, std::to_string(got2) + " vs " + std::to_string(want));
}

} // namespace

int main(int argc, const char **argv) {
  set_kernel_source_file(OPS_TEST_ACCESSOR_KERNELS);
  set_kernel_preamble("");
  ops_init(argc, argv, 1);
  Env e;
  e.block = ops_decl_block(2, "test_grid");
  int s00[] = {0, 0}, sx1[] = {0, 0, 1, 0};
  e.s00 = ops_decl_stencil(2, 1, s00, "0,0");
  int stride_x[] = {1, 0}, stride_y[] = {0, 1};
  e.s00x = ops_decl_strided_stencil(2, 1, s00, stride_x, "00_x");
  e.s00y = ops_decl_strided_stencil(2, 1, s00, stride_y, "00_y");
  e.s00x1 = ops_decl_strided_stencil(2, 2, sx1, stride_x, "00_p10_x");
  ops_partition("");
  caseReductions(e);
  caseLines(e);
  caseStates(e);
  std::printf("%d passed, %d failed\n", passes, failures);
  ops_exit();
  return failures ? 1 : 0;
}
