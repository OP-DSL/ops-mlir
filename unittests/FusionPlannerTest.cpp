//===- FusionPlannerTest.cpp - Unit tests for the loop fusion planner ----===//
//
// Standalone checks (no test framework, no JIT, no GPU) that run planFusion
// on synthetic LoopDescs and assert on the resulting groups.
//
//===----------------------------------------------------------------------===//

#include "runtime/FusionPlanner.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace ops_mlir;

namespace {

int failures = 0;

void check(bool ok, const std::string &name, const std::string &detail = "") {
  if (ok) {
    std::cout << "[PASS] " << name << "\n";
  } else {
    std::cout << "[FAIL] " << name << ": " << detail << "\n";
    ++failures;
  }
}

// Stencil offset tables must outlive the descriptors that point at them.
const int kPoint[2] = {0, 0};
const int kFive[10] = {0, 0, 1, 0, -1, 0, 0, 1, 0, -1};
const int kUnitStride[2] = {1, 1};
const int kBroadcast[2] = {1, 0};

StencilDesc pointStencil() {
  StencilDesc s{};
  s.dims = 2;
  s.points = 1;
  s.stencil = reinterpret_cast<std::uintptr_t>(kPoint);
  s.stride = reinterpret_cast<std::uintptr_t>(kUnitStride);
  return s;
}

StencilDesc fivePoint() {
  StencilDesc s{};
  s.dims = 2;
  s.points = 5;
  s.stencil = reinterpret_cast<std::uintptr_t>(kFive);
  s.stride = reinterpret_cast<std::uintptr_t>(kUnitStride);
  return s;
}

ArgDesc datArg(int dat, int acc, bool fivePt = false) {
  ArgDesc a{};
  a.argtype = OPS_ARG_DAT;
  a.acc = acc;
  a.dat.index = dat;
  a.stencil = fivePt ? fivePoint() : pointStencil();
  return a;
}

ArgDesc reductionArg() {
  ArgDesc a{};
  a.argtype = OPS_ARG_GBL;
  a.acc = OPS_MAX;
  return a;
}

ArgDesc globalArg() {
  ArgDesc a{};
  a.argtype = OPS_ARG_GBL;
  a.acc = OPS_READ;
  return a;
}

const std::vector<int64_t> kFull = {-1, 11, -1, 11};
const std::vector<int64_t> kInner = {0, 10, 0, 10};
const std::vector<int64_t> kLeft = {-1, 0, -1, 11};

LoopDesc loop(std::vector<ArgDesc> args, std::vector<int64_t> range = kInner,
              std::uintptr_t block = 1) {
  LoopDesc l;
  l.kernel_name = "k";
  l.block = block;
  l.dims = 2;
  l.range = std::move(range);
  l.args = std::move(args);
  return l;
}

std::string plan(const std::vector<LoopDesc> &q, FusionOptions o = {}) {
  return planFusion(q, o).digest();
}

FusionOptions unguarded() {
  FusionOptions o;
  o.allowGuarded = false;
  return o;
}

FusionOptions guarded(double ratio = 1.0) {
  FusionOptions o;
  o.allowGuarded = true;
  o.boxRatio = ratio;
  return o;
}

} // namespace

int main() {
  // a = f(b); c = g(a); d = h(c): point-wise chain, one group.
  {
    std::vector<LoopDesc> q = {
        loop({datArg(1, OPS_READ), datArg(2, OPS_WRITE)}),
        loop({datArg(2, OPS_READ), datArg(3, OPS_WRITE)}),
        loop({datArg(3, OPS_READ), datArg(4, OPS_WRITE)})};
    check(plan(q) == "0,1,2", "pointwise chain fuses", plan(q));
  }

  // Non-local RAW: 5-point read of a dat the previous loop wrote.
  {
    std::vector<LoopDesc> q = {
        loop({datArg(1, OPS_READ), datArg(2, OPS_WRITE)}),
        loop({datArg(2, OPS_READ, true), datArg(3, OPS_WRITE)})};
    check(plan(q) == "0|1", "stencil RAW stays split", plan(q));
  }

  // Non-local WAR: the Jacobi pair (Anew = lap(A); A = Anew).
  {
    std::vector<LoopDesc> q = {
        loop({datArg(1, OPS_READ, true), datArg(2, OPS_WRITE)}),
        loop({datArg(2, OPS_READ), datArg(1, OPS_WRITE)})};
    check(plan(q) == "0|1", "stencil WAR (Jacobi) stays split", plan(q));
  }

  // Reading a stencil of a dat nobody writes in the group is fine.
  {
    std::vector<LoopDesc> q = {
        loop({datArg(1, OPS_READ, true), datArg(2, OPS_WRITE)}),
        loop({datArg(1, OPS_READ, true), datArg(3, OPS_WRITE)})};
    check(plan(q) == "0,1", "two stencil readers of one dat fuse", plan(q));
  }

  // WAW and zero-offset RAW/WAR are point-local.
  {
    std::vector<LoopDesc> q = {
        loop({datArg(1, OPS_WRITE)}), loop({datArg(1, OPS_WRITE)}),
        loop({datArg(1, OPS_RW)}),
        loop({datArg(1, OPS_READ), datArg(2, OPS_WRITE)}),
        loop({datArg(2, OPS_READ), datArg(1, OPS_WRITE)})};
    check(plan(q) == "0,1,2,3,4", "WAW / zero-offset RAW+WAR fuse", plan(q));
  }

  // A reduction loop is a barrier on both sides.
  {
    std::vector<LoopDesc> q = {
        loop({datArg(1, OPS_WRITE)}),
        loop({datArg(1, OPS_READ), reductionArg()}),
        loop({datArg(2, OPS_WRITE)})};
    check(plan(q) == "0|1|2", "reduction loop is a barrier", plan(q));
  }

  // Read-only globals do not block fusion.
  {
    std::vector<LoopDesc> q = {
        loop({datArg(1, OPS_WRITE), globalArg()}),
        loop({datArg(2, OPS_WRITE), globalArg()})};
    check(plan(q) == "0,1", "read-only globals fuse", plan(q));
  }

  // Different block / dims never fuse.
  {
    std::vector<LoopDesc> q = {loop({datArg(1, OPS_WRITE)}, kInner, 1),
                               loop({datArg(2, OPS_WRITE)}, kInner, 2)};
    check(plan(q) == "0|1", "different blocks stay split", plan(q));
  }

  // Broadcast / strided stencils are not point-local.
  {
    ArgDesc strided = datArg(2, OPS_READ);
    strided.stencil.stride = reinterpret_cast<std::uintptr_t>(kBroadcast);
    std::vector<LoopDesc> q = {loop({datArg(2, OPS_WRITE)}),
                               loop({strided, datArg(3, OPS_WRITE)})};
    check(plan(q) == "0|1", "strided read after write stays split", plan(q));
  }

  // Different ranges: split unless guarded fusion is enabled.
  {
    std::vector<LoopDesc> q = {loop({datArg(1, OPS_WRITE)}, kFull),
                               loop({datArg(1, OPS_WRITE)}, kLeft)};
    check(plan(q, unguarded()) == "0|1", "different ranges split when guarded is off",
          plan(q, unguarded()));
    check(plan(q) == "0,1g", "guarded fusion is on by default", plan(q));
    check(plan(q, guarded()) == "0,1g", "different ranges fuse when guarded",
          plan(q, guarded()));
    auto p = planFusion(q, guarded());
    check(p.groups.size() == 1 && p.groups[0].guarded &&
              p.groups[0].range == kFull,
          "guarded group carries the bounding box");
  }

  // Laplace-style setup: full + four boundary strips, all writing one dat.
  {
    const std::vector<int64_t> bottom = {-1, 11, -1, 0}, top = {-1, 11, 10, 11},
                               left = {-1, 0, -1, 11}, right = {10, 11, -1, 11};
    std::vector<LoopDesc> q = {
        loop({datArg(1, OPS_WRITE)}, kFull), loop({datArg(1, OPS_WRITE)}, bottom),
        loop({datArg(1, OPS_WRITE)}, top),   loop({datArg(1, OPS_WRITE)}, left),
        loop({datArg(1, OPS_WRITE)}, right)};
    check(plan(q, guarded()) == "0,1,2,3,4g", "setup + boundary strips fuse",
          plan(q, guarded()));
  }

  // Thin loops far apart are not worth one full-grid launch.
  {
    const std::vector<int64_t> rowA = {0, 100, 0, 1}, rowB = {0, 100, 99, 100};
    std::vector<LoopDesc> q = {loop({datArg(1, OPS_WRITE)}, rowA),
                               loop({datArg(2, OPS_WRITE)}, rowB)};
    check(plan(q, guarded(1.0)) == "0|1", "bounding-box ratio splits",
          plan(q, guarded(1.0)));
    check(plan(q, guarded(100.0)) == "0,1g", "large ratio allows the merge",
          plan(q, guarded(100.0)));
  }

  // Legality is independent of range: RAW across points still splits.
  {
    std::vector<LoopDesc> q = {
        loop({datArg(1, OPS_WRITE)}, kFull),
        loop({datArg(1, OPS_READ, true), datArg(2, OPS_WRITE)}, kInner)};
    check(plan(q, guarded()) == "0|1", "guarded still respects RAW",
          plan(q, guarded()));
  }

  // A guarded member smaller than the bounding box is evaluated outside its
  // own range, so it may only read point-locally.
  {
    // 5-point consumer over the interior + a full-grid writer: the writer
    // grows the box past the consumer's range.
    std::vector<LoopDesc> q = {
        loop({datArg(1, OPS_READ, true), datArg(2, OPS_WRITE)}, kInner),
        loop({datArg(3, OPS_WRITE)}, kFull)};
    check(plan(q, guarded(4.0)) == "0|1", "stencil member cannot shrink inside a box",
          plan(q, guarded(4.0)));
    // ...but a stencil member that IS the box is fine.
    std::vector<LoopDesc> q2 = {
        loop({datArg(3, OPS_WRITE)}, kInner),
        loop({datArg(1, OPS_READ, true), datArg(2, OPS_WRITE)}, kFull)};
    check(plan(q2, guarded(4.0)) == "0,1g", "stencil member spanning the box fuses",
          plan(q2, guarded(4.0)));
    // A later, bigger loop must not retroactively strand an earlier stencil member.
    std::vector<LoopDesc> q3 = {
        loop({datArg(1, OPS_READ, true), datArg(2, OPS_WRITE)}, kInner),
        loop({datArg(3, OPS_WRITE)}, kInner),
        loop({datArg(4, OPS_WRITE)}, kFull)};
    check(plan(q3, guarded(4.0)) == "0,1|2", "growing the box re-validates members",
          plan(q3, guarded(4.0)));
  }

  // Group size cap.
  {
    std::vector<LoopDesc> q;
    for (int i = 0; i < 5; ++i)
      q.push_back(loop({datArg(i + 1, OPS_WRITE)}));
    FusionOptions o;
    o.maxGroupSize = 2;
    check(plan(q, o) == "0,1|2,3|4", "group size cap", plan(q, o));
  }

  // Fusion disabled: every loop alone.
  {
    std::vector<LoopDesc> q = {loop({datArg(1, OPS_WRITE)}),
                               loop({datArg(2, OPS_WRITE)})};
    FusionOptions o;
    o.enabled = false;
    check(plan(q, o) == "0|1", "disabled -> singletons", plan(q, o));
  }

  // Empty queue.
  check(plan({}) == "", "empty queue");

  // Every loop is placed exactly once, in order.
  {
    std::vector<LoopDesc> q = {
        loop({datArg(1, OPS_WRITE)}),
        loop({datArg(1, OPS_READ, true), datArg(2, OPS_WRITE)}),
        loop({datArg(2, OPS_READ), datArg(3, OPS_WRITE)}),
        loop({datArg(3, OPS_READ), reductionArg()}), loop({datArg(4, OPS_WRITE)})};
    auto p = planFusion(q, guarded());
    std::vector<std::size_t> flat;
    for (auto &g : p.groups)
      flat.insert(flat.end(), g.loops.begin(), g.loops.end());
    bool ok = flat.size() == q.size();
    for (std::size_t i = 0; ok && i < flat.size(); ++i)
      ok = flat[i] == i;
    check(ok, "each loop appears once, in order", p.digest());
  }

  if (failures == 0) {
    std::cout << "All FusionPlanner tests passed.\n";
    return 0;
  }
  std::cout << failures << " FusionPlanner test(s) failed.\n";
  return 1;
}
