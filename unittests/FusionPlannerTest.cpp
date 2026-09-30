//===- FusionPlannerTest.cpp - Unit tests for the loop fusion planner ----===//
//
// Standalone checks (no test framework, no JIT, no GPU) that run planFusion
// on synthetic LoopDescs and assert on the resulting groups.
//
//===----------------------------------------------------------------------===//

#include "runtime/FusionPlanner.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
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

  //===------------------------------------------------------------------===//
  // Reordering (DAG planner)
  //===------------------------------------------------------------------===//

  auto consecutive = [] {
    FusionOptions o;
    o.reorder = false;
    return o;
  };
  auto latest = [] {
    FusionOptions o;
    o.placement = FusionOptions::Placement::Latest;
    return o;
  };

  // Two interleaved chains, each link a stencil read of the previous output:
  //   a1 a2 b1 b2 a3 b3     (a_k reads a_{k-1} 5pt; same for b)
  // Adjacent grouping is forced into 4 kernels; the DAG planner runs the
  // independent chains side by side in 3.
  {
    std::vector<LoopDesc> q = {
        loop({datArg(1, OPS_READ), datArg(2, OPS_WRITE)}),        // a1
        loop({datArg(2, OPS_READ, true), datArg(3, OPS_WRITE)}),  // a2
        loop({datArg(11, OPS_READ), datArg(12, OPS_WRITE)}),      // b1
        loop({datArg(12, OPS_READ, true), datArg(13, OPS_WRITE)}),// b2
        loop({datArg(3, OPS_READ, true), datArg(4, OPS_WRITE)}),  // a3
        loop({datArg(13, OPS_READ, true), datArg(14, OPS_WRITE)}) // b3
    };
    check(plan(q, consecutive()) == "0|1,2|3,4|5", "consecutive planner: 4 kernels",
          plan(q, consecutive()));
    check(plan(q) == "0,2|1,3|4,5", "DAG planner: independent chains side by side",
          plan(q));
    check(plan(q, latest()) == "0|1,2|3,4|5", "latest placement matches adjacent grouping",
          plan(q, latest()));
    check(planFusion(q, consecutive()).numReordered() == 0, "consecutive planner never reorders");
    check(planFusion(q, FusionOptions()).numReordered() > 0, "DAG planner reports reordering");
  }

  // Footprints, not whole dats, define dependences: a stencil reader whose
  // reach does not touch what an earlier loop wrote is independent of it, so
  // a later loop can hoist past it.
  {
    const std::vector<int64_t> strip = {-1, 0, -1, 11}, inner = {3, 10, 0, 10};
    std::vector<LoopDesc> q = {
        loop({datArg(1, OPS_WRITE)}, strip),                        // L0 strip
        loop({datArg(1, OPS_READ, true), datArg(2, OPS_WRITE)}, inner), // L1 reads dat1 far from the strip
        loop({datArg(3, OPS_WRITE)}, strip)};                       // L2 strip
    check(plan(q, consecutive()) == "0|1|2", "consecutive: three kernels", plan(q, consecutive()));
    check(plan(q) == "0,2|1", "DAG: hoists L2 next to L0", plan(q));
    // The same read overlapping the strip IS a dependence.
    const std::vector<int64_t> near = {0, 10, 0, 10};
    std::vector<LoopDesc> q2 = {
        loop({datArg(1, OPS_WRITE)}, strip),
        loop({datArg(1, OPS_READ, true), datArg(2, OPS_WRITE)}, near),
        loop({datArg(3, OPS_WRITE)}, strip)};
    check(plan(q2) == "0,2|1", "DAG: dependent reader still ordered after L0", plan(q2));
    auto p2 = planFusion(q2, FusionOptions());
    check(p2.groups.size() == 2 && p2.groups[0].loops.front() == 0 &&
              p2.groups[1].loops.front() == 1,
          "DAG: reader runs after its producer");
  }

  // A loop never moves across a reduction barrier.
  {
    std::vector<LoopDesc> q = {loop({datArg(1, OPS_WRITE)}),
                               loop({datArg(1, OPS_READ), reductionArg()}),
                               loop({datArg(2, OPS_WRITE)})};
    check(plan(q) == "0|1|2", "DAG: reduction is a barrier for reordering", plan(q));
  }

  // A loop may not jump ahead of a loop it depends on, even one it could
  // otherwise fuse with: c2 needs b (point), b needs a (stencil).
  {
    std::vector<LoopDesc> q = {
        loop({datArg(1, OPS_WRITE)}),                                // 0: a
        loop({datArg(1, OPS_READ, true), datArg(2, OPS_WRITE)}),     // 1: b = f(a)
        loop({datArg(2, OPS_READ), datArg(3, OPS_WRITE)})};          // 2: c = g(b)
    check(plan(q) == "0|1,2", "DAG: dependent loop stays behind its producer", plan(q));
  }

  //===------------------------------------------------------------------===//
  // Randomized oracle: a plan must compute exactly what program order does
  //===------------------------------------------------------------------===//
  //
  // Random 1D programs over a few dats. Reference: run the loops in order.
  // Candidate: run the plan's groups in order; inside a group visit each box
  // point and, at each, every member whose range contains it, in member
  // order, writing in place. Any planner bug (a wrongly fused WAR/RAW, a move
  // across a dependence, a bad bounding box) shows up as different data.
  {
    static const int kOffsetSets[][4] = {{0, 0, 0, 0}, {-1, 0, 1, 3}, {1, 0, 0, 2}, {-1, 0, 0, 2}};
    static const int kNumPts[] = {1, 3, 2, 2};
    static const int kOne[1] = {1};
    const int kGrid = 14, kBase = 2; // dat cell index = x + kBase
    const int64_t kMod = 1000003;

    struct Arg { int dat, acc, st; };
    struct Prog { std::vector<std::vector<Arg>> loops; std::vector<std::pair<int, int>> ranges; };

    std::mt19937 rng(12345);
    auto rnd = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };

    auto toQueue = [&](const Prog &pr) {
      std::vector<LoopDesc> q;
      for (std::size_t i = 0; i < pr.loops.size(); ++i) {
        LoopDesc l;
        l.kernel_name = "k";
        l.block = 1;
        l.dims = 1;
        l.range = {pr.ranges[i].first, pr.ranges[i].second};
        for (const Arg &a : pr.loops[i]) {
          ArgDesc d{};
          d.argtype = OPS_ARG_DAT;
          d.acc = a.acc;
          d.dat.index = a.dat;
          d.stencil.dims = 1;
          d.stencil.points = kNumPts[a.st];
          d.stencil.stencil = reinterpret_cast<std::uintptr_t>(kOffsetSets[a.st]);
          d.stencil.stride = reinterpret_cast<std::uintptr_t>(kOne);
          l.args.push_back(d);
        }
        q.push_back(std::move(l));
      }
      return q;
    };

    using Grid = std::vector<std::vector<int64_t>>;
    auto execAt = [&](const Prog &pr, std::size_t li, int x, Grid &g) {
      int64_t v = 7 * (li + 1);
      for (const Arg &a : pr.loops[li]) {
        if (a.acc == OPS_WRITE)
          continue;
        for (int k = 0; k < kNumPts[a.st]; ++k)
          v = (v * 31 + g[a.dat][x + kOffsetSets[a.st][k] + kBase]) % kMod;
      }
      for (const Arg &a : pr.loops[li])
        if (a.acc != OPS_READ)
          g[a.dat][x + kBase] = (v + 17 * a.dat) % kMod;
    };

    auto freshGrid = [&] {
      Grid g(4, std::vector<int64_t>(kGrid + 2 * kBase));
      for (int d = 0; d < 4; ++d)
        for (int x = 0; x < kGrid + 2 * kBase; ++x)
          g[d][x] = (d * 101 + x * 13 + 5) % kMod;
      return g;
    };

    int bad = 0, reordered = 0, fusedLoops = 0, total = 0;
    std::string firstBad;
    for (int trial = 0; trial < 4000; ++trial) {
      Prog pr;
      int nloops = rnd(2, 12);
      for (int i = 0; i < nloops; ++i) {
        std::vector<Arg> args;
        int nargs = rnd(1, 3);
        std::map<int, bool> used;
        bool writes = false;
        for (int a = 0; a < nargs; ++a) {
          int dat = rnd(0, 3);
          if (used[dat])
            continue;
          used[dat] = true;
          int mode = rnd(0, 2);
          int acc = mode == 0 ? OPS_READ : mode == 1 ? OPS_WRITE : OPS_RW;
          // OPS requires a dat written in a loop to be accessed point-locally.
          int st = acc == OPS_READ ? rnd(0, 3) : 0;
          writes |= acc != OPS_READ;
          args.push_back({dat, acc, st});
        }
        if (!writes)
          args.push_back({rnd(0, 3), OPS_WRITE, 0});
        // A dat read through a stencil must not also be written by this loop.
        std::map<int, bool> wr;
        for (const Arg &a : args) if (a.acc != OPS_READ) wr[a.dat] = true;
        for (Arg &a : args) if (a.acc == OPS_READ && wr[a.dat]) a.st = 0;
        // Drop duplicate dats the fallback write may have created.
        std::vector<Arg> uniq; std::map<int, int> seen;
        for (const Arg &a : args) if (seen[a.dat]++ == 0) uniq.push_back(a);
        pr.loops.push_back(uniq);
        int lo = rnd(0, kGrid - 2), hi = rnd(lo + 1, kGrid);
        pr.ranges.push_back({lo, hi});
      }
      std::vector<LoopDesc> q = toQueue(pr);

      Grid ref = freshGrid();
      for (std::size_t li = 0; li < q.size(); ++li)
        for (int x = pr.ranges[li].first; x < pr.ranges[li].second; ++x)
          execAt(pr, li, x, ref);

      for (int variant = 0; variant < 6; ++variant) {
        FusionOptions o;
        o.reorder = variant % 2 == 0;
        o.allowGuarded = variant % 3 != 0;
        o.boxRatio = variant < 3 ? 1.0 : 50.0;
        o.maxGroupSize = variant == 5 ? 3 : 8;
        o.placement = variant == 4 ? FusionOptions::Placement::Latest
                                   : FusionOptions::Placement::Earliest;
        FusionPlan plan = planFusion(q, o);

        // Every loop exactly once.
        std::vector<int> seenLoop(q.size(), 0);
        bool wellFormed = true;
        for (const FusedGroup &g : plan.groups) {
          for (std::size_t k = 0; k < g.loops.size(); ++k) {
            wellFormed &= ++seenLoop[g.loops[k]] == 1;
            wellFormed &= k == 0 || g.loops[k - 1] < g.loops[k];
          }
        }
        for (int c : seenLoop) wellFormed &= c == 1;

        Grid got = freshGrid();
        for (const FusedGroup &g : plan.groups) {
          int lo = g.range[0], hi = g.range[1];
          for (int x = lo; x < hi; ++x)
            for (std::size_t li : g.loops)
              if (x >= pr.ranges[li].first && x < pr.ranges[li].second)
                execAt(pr, li, x, got);
          // The group's box must cover every member.
          for (std::size_t li : g.loops)
            wellFormed &= pr.ranges[li].first >= lo && pr.ranges[li].second <= hi;
          if (g.loops.size() > 1) fusedLoops += static_cast<int>(g.loops.size());
        }
        ++total;
        if (plan.numReordered() > 0) ++reordered;
        if (!wellFormed || got != ref) {
          if (!bad++)
            firstBad = "trial " + std::to_string(trial) + " variant " + std::to_string(variant) +
                       " plan " + plan.digest() + (wellFormed ? "" : " (malformed)");
        }
      }
    }
    check(bad == 0, "randomized oracle: plan == program order (" + std::to_string(total) + " plans)",
          std::to_string(bad) + " mismatches; first: " + firstBad);
    check(reordered > 100, "randomized oracle exercises reordering",
          "only " + std::to_string(reordered) + " reordered plans");
    check(fusedLoops > 1000, "randomized oracle exercises fusion",
          "only " + std::to_string(fusedLoops) + " fused loop slots");
  }

  if (failures == 0) {
    std::cout << "All FusionPlanner tests passed.\n";
    return 0;
  }
  std::cout << failures << " FusionPlanner test(s) failed.\n";
  return 1;
}
