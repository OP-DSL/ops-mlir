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

} // namespace

FusionPlan planFusion(const std::vector<LoopDesc> &queue,
                      const FusionOptions &options) {
  FusionPlan plan;

  DatUses groupUses;
  double memberVolume = 0.0; // sum of member volumes of the open group
  bool open = false;
  // Per member of the open group: range, and whether it reads any dat through
  // a non-point stencil. Needed to validate guarded groups (below).
  struct Member {
    std::vector<int64_t> range;
    bool nonLocalRead;
  };
  std::vector<Member> members;

  auto closeGroup = [&] { open = false; };
  auto startGroup = [&](std::size_t i) {
    FusedGroup g;
    g.loops.push_back(i);
    g.range = queue[i].range;
    plan.groups.push_back(std::move(g));
    groupUses = usesOf(queue[i]);
    memberVolume = volume(queue[i].range);
    members = {{queue[i].range, hasNonLocalRead(groupUses)}};
    open = true;
  };

  for (std::size_t i = 0; i < queue.size(); ++i) {
    const LoopDesc &loop = queue[i];

    // A loop with a reduction is never fused with its neighbours.
    if (!options.enabled || hasReduction(loop)) {
      closeGroup();
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
              group.loops.size() < options.maxGroupSize &&
              !hasReduction(first);

    DatUses nextUses;
    if (ok) {
      nextUses = usesOf(loop);
      ok = compatible(groupUses, nextUses);
    }

    bool guarded = group.guarded;
    std::vector<int64_t> box = group.range;
    if (ok && loop.range != group.range) {
      if (!options.allowGuarded) {
        ok = false;
      } else {
        box = boundingBox(group.range, loop.range);
        double total = memberVolume + volume(loop.range);
        ok = volume(box) <= options.boxRatio * total;
        guarded = true;
        // A guarded member is evaluated (and its inputs read) at every point
        // of the bounding box, not only inside its own range. Point-local
        // reads are always inside the allocation; a stencil read could fall
        // outside it. So every member smaller than the box must be
        // point-local.
        bool nextNonLocal = hasNonLocalRead(nextUses);
        if (ok && nextNonLocal && loop.range != box)
          ok = false;
        for (const Member &m : members)
          if (ok && m.nonLocalRead && m.range != box)
            ok = false;
      }
    }

    if (!ok) {
      startGroup(i);
      continue;
    }

    group.loops.push_back(i);
    group.range = std::move(box);
    group.guarded = guarded;
    memberVolume += volume(loop.range);
    members.push_back({loop.range, hasNonLocalRead(nextUses)});
    mergeUses(groupUses, nextUses);
  }

  return plan;
}

} // namespace ops_mlir
