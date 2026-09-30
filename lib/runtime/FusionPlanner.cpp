//===- FusionPlanner.cpp - Dependence-aware grouping of queued loops -----===//
//
// Part of OPS-MLIR Project
//
// This file is distributed under the MIT License.
// See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

#include "runtime/FusionPlanner.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <set>

namespace ops_mlir {

FusionOptions FusionOptions::fromEnv() {
  FusionOptions opts;
  if (const char *v = std::getenv("OPS_MLIR_FUSION"))
    opts.enabled = std::atoi(v) != 0;
  if (const char *v = std::getenv("OPS_MLIR_FUSION_MAX")) {
    long n = std::atol(v);
    if (n >= 1)
      opts.maxGroupSize = static_cast<std::size_t>(n);
  }
  if (const char *v = std::getenv("OPS_MLIR_FUSION_GUARDED"))
    opts.allowGuarded = std::atoi(v) != 0;
  if (const char *v = std::getenv("OPS_MLIR_FUSION_REORDER"))
    opts.reorder = std::atoi(v) != 0;
  if (const char *v = std::getenv("OPS_MLIR_FUSION_PLACEMENT"))
    opts.placement = std::string(v) == "latest"
                         ? FusionOptions::Placement::Latest
                         : FusionOptions::Placement::Earliest;
  if (const char *v = std::getenv("OPS_MLIR_FUSION_BOX_RATIO")) {
    double r = std::atof(v);
    if (r > 0.0)
      opts.boxRatio = r;
  }
  return opts;
}

std::string FusionPlan::digest() const {
  std::string out;
  for (std::size_t g = 0; g < groups.size(); ++g) {
    if (g)
      out += '|';
    for (std::size_t k = 0; k < groups[g].loops.size(); ++k) {
      if (k)
        out += ',';
      out += std::to_string(groups[g].loops[k]);
    }
    if (groups[g].guarded)
      out += 'g';
  }
  return out;
}

std::size_t FusionPlan::numReordered() const {
  std::size_t position = 0, moved = 0;
  for (const FusedGroup &g : groups)
    for (std::size_t l : g.loops)
      moved += (l != position++);
  return moved;
}

bool isPointLocal(const StencilDesc &st) {
  if (st.stencil == 0 || st.points == 0)
    return true; // matches the lowering: no offsets means the zero offset
  const int *offsets = reinterpret_cast<const int *>(st.stencil);
  for (int i = 0; i < st.dims * st.points; ++i)
    if (offsets[i] != 0)
      return false;
  if (st.stride) {
    const int *stride = reinterpret_cast<const int *>(st.stride);
    for (int d = 0; d < st.dims; ++d)
      if (stride[d] != 1)
        return false; // multigrid / broadcast access
  }
  return true;
}

bool hasReduction(const LoopDesc &loop) {
  for (const ArgDesc &arg : loop.args)
    if (arg.argtype == OPS_ARG_GBL && arg.acc != OPS_READ)
      return true;
  return false;
}

namespace {

// How a loop or a group touches one dat.
struct DatUse {
  bool reads = false;
  bool readsNonLocal = false;
  bool writes = false;
};

using DatUses = std::map<int, DatUse>; // keyed by ops_dat index

bool isWrite(int acc) {
  return acc == OPS_WRITE || acc == OPS_RW || acc == OPS_INC;
}
bool isRead(int acc) { return acc == OPS_READ || acc == OPS_RW; }

DatUses usesOf(const LoopDesc &loop) {
  DatUses uses;
  for (const ArgDesc &arg : loop.args) {
    if (arg.argtype != OPS_ARG_DAT)
      continue;
    DatUse &u = uses[arg.dat.index];
    if (isRead(arg.acc)) {
      u.reads = true;
      if (!isPointLocal(arg.stencil))
        u.readsNonLocal = true;
    }
    if (isWrite(arg.acc)) {
      u.writes = true;
      // A write through a non-point stencil has no defined meaning here;
      // treat it like a non-local read so it can never be reordered.
      if (!isPointLocal(arg.stencil))
        u.readsNonLocal = true;
    }
  }
  return uses;
}

// Legality of appending a loop with uses `next` to a group with uses `group`.
bool compatible(const DatUses &group, const DatUses &next) {
  for (const auto &[dat, n] : next) {
    auto it = group.find(dat);
    if (it == group.end())
      continue;
    const DatUse &g = it->second;
    if (g.writes && n.reads && n.readsNonLocal)
      return false; // RAW across points
    if (g.readsNonLocal && n.writes)
      return false; // WAR across points
    if (g.writes && n.writes && (g.readsNonLocal || n.readsNonLocal))
      return false; // odd non-point write; keep ordered
  }
  return true;
}

void mergeUses(DatUses &group, const DatUses &next) {
  for (const auto &[dat, n] : next) {
    DatUse &g = group[dat];
    g.reads |= n.reads;
    g.readsNonLocal |= n.readsNonLocal;
    g.writes |= n.writes;
  }
}

bool hasNonLocalRead(const DatUses &uses) {
  for (const auto &[dat, u] : uses)
    if (u.readsNonLocal)
      return true;
  return false;
}

double volume(const std::vector<int64_t> &range) {
  double v = 1.0;
  for (std::size_t d = 0; d + 1 < range.size(); d += 2)
    v *= static_cast<double>(std::max<int64_t>(0, range[d + 1] - range[d]));
  return v;
}

std::vector<int64_t> boundingBox(const std::vector<int64_t> &a,
                                 const std::vector<int64_t> &b) {
  std::vector<int64_t> box = a;
  for (std::size_t d = 0; d + 1 < box.size(); d += 2) {
    box[d] = std::min(a[d], b[d]);
    box[d + 1] = std::max(a[d + 1], b[d + 1]);
  }
  return box;
}

// Decides whether a loop with range `loopRange` can join a group whose state
// is `g`, and what the group's box/guarded flag become. Shared by both
// planners. Different ranges need guarded fusion enabled, a bounding box not
// much larger than the members' volumes, and every member smaller than the box
// to read point-locally (a guarded member is evaluated at every box point, so
// a stencil read could leave the allocation).
struct Member {
  std::vector<int64_t> range;
  bool nonLocalRead;
};

struct GroupState {
  std::vector<int64_t> range;
  double memberVolume = 0.0;
  bool guarded = false;
  std::vector<Member> members;
};

bool rangeJoin(const GroupState &g, const std::vector<int64_t> &loopRange,
               bool loopNonLocalRead, const FusionOptions &options,
               std::vector<int64_t> &boxOut, bool &guardedOut) {
  boxOut = g.range;
  guardedOut = g.guarded;
  if (loopRange == g.range)
    return true;
  if (!options.allowGuarded)
    return false;
  boxOut = boundingBox(g.range, loopRange);
  guardedOut = true;
  if (volume(boxOut) > options.boxRatio * (g.memberVolume + volume(loopRange)))
    return false;
  if (loopNonLocalRead && loopRange != boxOut)
    return false;
  for (const Member &m : g.members)
    if (m.nonLocalRead && m.range != boxOut)
      return false;
  return true;
}

void addMember(GroupState &g, const std::vector<int64_t> &loopRange,
               bool nonLocalRead, std::vector<int64_t> box, bool guarded) {
  g.range = std::move(box);
  g.guarded = guarded;
  g.memberVolume += volume(loopRange);
  g.members.push_back({loopRange, nonLocalRead});
}

//===----------------------------------------------------------------------===//
// Consecutive planner
//===----------------------------------------------------------------------===//

FusionPlan planConsecutive(const std::vector<LoopDesc> &queue,
                           const FusionOptions &options) {
  FusionPlan plan;
  DatUses groupUses;
  GroupState state;
  bool open = false;

  auto startGroup = [&](std::size_t i) {
    FusedGroup g;
    g.loops.push_back(i);
    g.range = queue[i].range;
    plan.groups.push_back(std::move(g));
    groupUses = usesOf(queue[i]);
    state = GroupState();
    state.range = queue[i].range;
    state.memberVolume = volume(queue[i].range);
    state.members = {{queue[i].range, hasNonLocalRead(groupUses)}};
    open = true;
  };

  for (std::size_t i = 0; i < queue.size(); ++i) {
    const LoopDesc &loop = queue[i];

    // A loop with a reduction is never fused with its neighbours.
    if (!options.enabled || hasReduction(loop)) {
      open = false;
      FusedGroup g;
      g.loops.push_back(i);
      g.range = loop.range;
      plan.groups.push_back(std::move(g));
      continue;
    }
    if (!open) {
      startGroup(i);
      continue;
    }

    FusedGroup &group = plan.groups.back();
    const LoopDesc &first = queue[group.loops.front()];
    bool ok = loop.block == first.block && loop.dims == first.dims &&
              group.loops.size() < options.maxGroupSize && !hasReduction(first);

    DatUses nextUses;
    std::vector<int64_t> box;
    bool guarded = false;
    if (ok) {
      nextUses = usesOf(loop);
      ok = compatible(groupUses, nextUses) &&
           rangeJoin(state, loop.range, hasNonLocalRead(nextUses), options, box,
                     guarded);
    }
    if (!ok) {
      startGroup(i);
      continue;
    }
    group.loops.push_back(i);
    group.range = box;
    group.guarded = guarded;
    addMember(state, loop.range, hasNonLocalRead(nextUses), box, guarded);
    mergeUses(groupUses, nextUses);
  }
  return plan;
}

//===----------------------------------------------------------------------===//
// DAG planner
//===----------------------------------------------------------------------===//

using Range = std::vector<int64_t>;

bool nonEmpty(const Range &r) {
  for (std::size_t d = 0; d + 1 < r.size(); d += 2)
    if (r[d] >= r[d + 1])
      return false;
  return true;
}

bool overlaps(const Range &a, const Range &b) {
  if (!nonEmpty(a) || !nonEmpty(b))
    return false;
  std::size_t n = std::min(a.size(), b.size());
  for (std::size_t d = 0; d + 1 < n; d += 2)
    if (!(a[d] < b[d + 1] && b[d] < a[d + 1]))
      return false;
  return true;
}

constexpr int64_t kUnbounded = int64_t(1) << 40;

// Points a loop may touch through `arg`: its range expanded by the stencil.
// Multigrid / broadcast strides are treated as touching everything.
Range footprint(const LoopDesc &loop, const ArgDesc &arg) {
  const StencilDesc &st = arg.stencil;
  Range r = loop.range;
  if (st.stencil == 0 || st.points == 0)
    return r;
  const int *offsets = reinterpret_cast<const int *>(st.stencil);
  const int *stride = reinterpret_cast<const int *>(st.stride);
  for (int d = 0; d < st.dims && 2 * d + 1 < static_cast<int>(r.size()); ++d) {
    if (stride && stride[d] != 1) {
      r[2 * d] = -kUnbounded;
      r[2 * d + 1] = kUnbounded;
      continue;
    }
    int lo = offsets[d], hi = offsets[d];
    for (int p = 1; p < st.points; ++p) {
      lo = std::min(lo, offsets[p * st.dims + d]);
      hi = std::max(hi, offsets[p * st.dims + d]);
    }
    r[2 * d] += lo;
    r[2 * d + 1] += hi;
  }
  return r;
}

Range unionBox(const Range &a, const Range &b) {
  if (a.empty())
    return b;
  return boundingBox(a, b);
}

// What one loop does to one dat.
struct DatAccess {
  bool reads = false, writes = false;
  bool readNonLocal = false, writeNonLocal = false;
  Range readBox, writeBox;
};

std::map<int, DatAccess> accessesOf(const LoopDesc &loop) {
  std::map<int, DatAccess> out;
  for (const ArgDesc &arg : loop.args) {
    if (arg.argtype != OPS_ARG_DAT)
      continue;
    DatAccess &a = out[arg.dat.index];
    bool local = isPointLocal(arg.stencil);
    Range fp = footprint(loop, arg);
    if (isRead(arg.acc)) {
      a.reads = true;
      a.readNonLocal |= !local;
      a.readBox = unionBox(a.readBox, fp);
    }
    if (isWrite(arg.acc)) {
      a.writes = true;
      a.writeNonLocal |= !local;
      a.writeBox = unionBox(a.writeBox, fp);
    }
  }
  return out;
}

struct Pred {
  std::size_t from;
  bool fusable;
};

// One group being built by the DAG planner.
struct DagGroup {
  std::vector<std::size_t> loops;
  GroupState state;
  std::vector<int> succ; // groups that must run after this one
  bool barrier = false;
};

FusionPlan planDag(const std::vector<LoopDesc> &queue,
                   const FusionOptions &options) {
  const std::size_t n = queue.size();

  // --- dependence edges ----------------------------------------------------
  std::vector<std::map<int, DatAccess>> acc(n);
  std::vector<std::vector<Pred>> preds(n);
  std::map<int, std::vector<std::size_t>> history; // dat -> loops touching it
  for (std::size_t j = 0; j < n; ++j) {
    acc[j] = accessesOf(queue[j]);
    std::map<std::size_t, bool> edge; // pred -> all conflicts fusable
    for (const auto &[dat, aj] : acc[j]) {
      for (std::size_t i : history[dat]) {
        const DatAccess &ai = acc[i].at(dat);
        bool raw = ai.writes && aj.reads && overlaps(ai.writeBox, aj.readBox);
        bool war = ai.reads && aj.writes && overlaps(ai.readBox, aj.writeBox);
        bool waw = ai.writes && aj.writes && overlaps(ai.writeBox, aj.writeBox);
        if (!raw && !war && !waw)
          continue;
        bool fusable = true;
        if (raw && aj.readNonLocal)
          fusable = false; // j reads across points what i wrote
        if (war && ai.readNonLocal)
          fusable = false; // i read across points what j overwrites
        if (ai.writeNonLocal || aj.writeNonLocal)
          fusable = false;
        auto it = edge.find(i);
        if (it == edge.end())
          edge[i] = fusable;
        else
          it->second = it->second && fusable;
      }
      history[dat].push_back(j);
    }
    for (const auto &[i, fusable] : edge)
      preds[j].push_back({i, fusable});
  }

  // --- placement -----------------------------------------------------------
  std::vector<DagGroup> groups;
  std::vector<int> groupOf(n, -1);
  int lastBarrier = -1;
  std::size_t firstJoinable = 0; // groups before a barrier are sealed

  auto addEdge = [&](int from, int to) {
    if (from == to)
      return;
    auto &s = groups[from].succ;
    if (std::find(s.begin(), s.end(), to) == s.end())
      s.push_back(to);
  };
  auto reaches = [&](int from, int to) {
    std::vector<int> stack = {from};
    std::vector<char> seen(groups.size(), 0);
    seen[from] = 1;
    while (!stack.empty()) {
      int g = stack.back();
      stack.pop_back();
      if (g == to)
        return true;
      for (int nx : groups[g].succ)
        if (!seen[nx]) {
          seen[nx] = 1;
          stack.push_back(nx);
        }
    }
    return false;
  };
  auto newGroup = [&](std::size_t j) {
    DagGroup g;
    g.loops.push_back(j);
    g.state.range = queue[j].range;
    g.state.memberVolume = volume(queue[j].range);
    bool nonLocal = false;
    for (const auto &[dat, a] : acc[j])
      nonLocal |= a.readNonLocal;
    g.state.members = {{queue[j].range, nonLocal}};
    groups.push_back(std::move(g));
    return static_cast<int>(groups.size()) - 1;
  };

  for (std::size_t j = 0; j < n; ++j) {
    const LoopDesc &loop = queue[j];
    bool nonLocalRead = false;
    for (const auto &[dat, a] : acc[j])
      nonLocalRead |= a.readNonLocal;

    if (hasReduction(loop)) {
      // Barrier: nothing moves across it.
      int b = newGroup(j);
      groups[b].barrier = true;
      for (int h = 0; h < b; ++h)
        addEdge(h, b);
      groupOf[j] = b;
      lastBarrier = b;
      firstJoinable = groups.size();
      continue;
    }

    int chosen = -1;
    std::vector<int64_t> chosenBox;
    bool chosenGuarded = false;
    for (std::size_t g = firstJoinable; g < groups.size(); ++g) {
      const DagGroup &grp = groups[g];
      const LoopDesc &first = queue[grp.loops.front()];
      if (grp.barrier || grp.loops.size() >= options.maxGroupSize ||
          loop.block != first.block || loop.dims != first.dims)
        continue;
      bool ok = true;
      for (const Pred &p : preds[j]) {
        int h = groupOf[p.from];
        if (h == static_cast<int>(g)) {
          if (!p.fusable) {
            ok = false;
            break;
          }
        } else if (reaches(static_cast<int>(g), h)) {
          ok = false; // would put j before something that must run before it
          break;
        }
      }
      if (!ok)
        continue;
      std::vector<int64_t> box;
      bool guarded = false;
      if (!rangeJoin(grp.state, loop.range, nonLocalRead, options, box, guarded))
        continue;
      chosen = static_cast<int>(g);
      chosenBox = std::move(box);
      chosenGuarded = guarded;
      if (options.placement == FusionOptions::Placement::Earliest)
        break; // otherwise keep scanning for the latest
    }

    if (chosen >= 0) {
      DagGroup &g = groups[chosen];
      g.loops.push_back(j);
      addMember(g.state, loop.range, nonLocalRead, chosenBox, chosenGuarded);
    } else {
      chosen = newGroup(j);
    }
    groupOf[j] = chosen;
    for (const Pred &p : preds[j])
      addEdge(groupOf[p.from], chosen);
    if (lastBarrier >= 0)
      addEdge(lastBarrier, chosen);
  }

  // --- emission: topological order, ties broken by program order ------------
  std::vector<int> indegree(groups.size(), 0);
  for (const DagGroup &g : groups)
    for (int s : g.succ)
      ++indegree[s];
  std::set<std::pair<std::size_t, int>> ready; // (first loop, group)
  for (std::size_t g = 0; g < groups.size(); ++g)
    if (indegree[g] == 0)
      ready.insert({groups[g].loops.front(), static_cast<int>(g)});

  FusionPlan plan;
  while (!ready.empty()) {
    int g = ready.begin()->second;
    ready.erase(ready.begin());
    FusedGroup out;
    out.loops = groups[g].loops;
    out.guarded = groups[g].state.guarded;
    out.range = groups[g].state.range;
    plan.groups.push_back(std::move(out));
    for (int s : groups[g].succ)
      if (--indegree[s] == 0)
        ready.insert({groups[s].loops.front(), s});
  }
  // A cycle in the group graph would strand groups (and their loops). The
  // placement checks rule that out; if it ever happens, fall back to the
  // always-correct adjacent grouping rather than drop work.
  if (plan.groups.size() != groups.size())
    return planConsecutive(queue, options);
  return plan;
}

} // namespace

FusionPlan planFusion(const std::vector<LoopDesc> &queue,
                      const FusionOptions &options) {
  if (!options.enabled || !options.reorder)
    return planConsecutive(queue, options);
  return planDag(queue, options);
}

} // namespace ops_mlir
