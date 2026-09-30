//===- FusionPlanner.h - Dependence-aware grouping of queued loops -------===//
//
// Part of OPS-MLIR Project
//
// This file is distributed under the MIT License.
// See LICENSE.txt for details.
//
// Decides which queued par_loops run together as a single generated kernel,
// and in what order the kernels run. Pure C++ over LoopDesc: no MLIR, no GPU,
// unit-testable.
//
// Two planners share the legality rules below:
//   * consecutive (OPS_MLIR_FUSION_REORDER=0): only adjacent loops fuse;
//   * DAG (default): a dependence graph over the whole queue lets independent
//     loops be hoisted next to the loops they can fuse with, so a loop that
//     cannot join its neighbours may still join an earlier kernel.
//
// DAG planner. Between loops i < j that touch the same dat there is an edge
// when their footprints overlap (write range vs. read range expanded by the
// stencil, for RAW; likewise WAR, WAW). Loops joined by no path may run in
// either order. An edge is *fusable* when the value only moves within a
// point (RAW: j reads D at the zero offset; WAR: i read D at the zero
// offset). Loops are placed, in program order, into an existing group when
// every edge to a member is fusable and the groups stay acyclic (so no loop
// is moved across one it depends on); kernels are emitted in a topological
// order of the groups.
//
// Fusing loop j into a group that already holds loop i (i before j) executes
// both at each point p, in order, instead of running i over its whole range
// and then j. That is equivalent iff no value flows between different points:
//   * i writes D and j reads D at a non-zero offset  (RAW across points)
//   * i reads D at a non-zero offset and j writes D  (WAR across points)
// are the only ways it can, so those pairs stay separate. This holds for any
// pair of ranges: a point outside a member's range simply skips that member
// (a "guarded" group), and reads of a dat that member would have written see
// the old value, exactly as in sequential execution.
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_RUNTIME_FUSION_PLANNER_H
#define OPS_MLIR_RUNTIME_FUSION_PLANNER_H

#include "runtime/Core.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ops_mlir {

struct FusionOptions {
  bool enabled = true;
  // Upper bound on loops per group (register pressure / kernel size).
  std::size_t maxGroupSize = 8;
  // Fuse loops with different ranges into one guarded launch over the
  // bounding box.
  bool allowGuarded = true;
  // Build a dependence DAG and allow loops to move past independent ones.
  bool reorder = true;
  // Which legal group a loop joins when the DAG planner has a choice:
  // the earliest one (shortest dependence chains first; most fusion) or the
  // latest (keeps loops near their program-order neighbours).
  enum class Placement { Earliest, Latest };
  Placement placement = Placement::Earliest;
  // A guarded group is only formed if
  //   volume(bounding box) <= boxRatio * sum(member volumes)
  // so a thin boundary loop does not drag a full-grid launch along.
  double boxRatio = 1.0;

  /// Reads OPS_MLIR_FUSION, OPS_MLIR_FUSION_MAX, OPS_MLIR_FUSION_GUARDED,
  /// OPS_MLIR_FUSION_BOX_RATIO, OPS_MLIR_FUSION_REORDER and
  /// OPS_MLIR_FUSION_PLACEMENT (earliest|latest) on top of the defaults.
  static FusionOptions fromEnv();
};

struct FusedGroup {
  std::vector<std::size_t> loops; // indices into the queue, ascending (also
                                  // a valid order to run them in)
  bool guarded = false;           // members have differing ranges
  std::vector<int64_t> range;     // bounding box, OPS order [lo0,hi0,lo1,hi1..]
};

struct FusionPlan {
  std::vector<FusedGroup> groups; // in execution order

  /// DAG planner diagnostics: for every loop that had to start a new kernel,
  /// why each candidate kernel was rejected (a loop can count under several
  /// reasons). Keys: "first" (nothing to join), "dependence" (a non-point-
  /// local RAW/WAR/WAW edge to a member), "order" (joining would put it before
  /// a kernel it depends on), "range" (bounding-box / guard rule),
  /// "size" (group size cap), "block" (different block or dimension).
  std::map<std::string, std::size_t> blockedBy;

  /// Number of loops that execute in a different position relative to the
  /// others than program order (0 for the consecutive planner).
  std::size_t numReordered() const;

  /// Compact text form ("0,1,2|3|4,5"), part of the compiled-module cache key.
  std::string digest() const;
};

/// Memory traffic model in the style of OPS (extent x element size, read and
/// write counted separately, stencil neighbours assumed cached).
struct TrafficEstimate {
  double unfusedBytes = 0; // every loop on its own
  double fusedBytes = 0;   // per kernel: each distinct dat loaded at most once
                           // (not at all if a member writes it first) and
                           // stored at most once
};
TrafficEstimate estimateTraffic(const std::vector<LoopDesc> &queue,
                                const FusionPlan &plan);

/// Groups the queue. Every loop lands in exactly one group. With reordering
/// off the groups are runs of adjacent loops; with it on they may interleave
/// but every dependence is respected.
FusionPlan planFusion(const std::vector<LoopDesc> &queue,
                      const FusionOptions &options);

/// True if every point of the stencil is the zero offset with unit stride,
/// i.e. the access only touches the iteration point.
bool isPointLocal(const StencilDesc &stencil);

/// True if the loop has a reduction (a non-read-only global): those loops
/// are never fused.
bool hasReduction(const LoopDesc &loop);

} // namespace ops_mlir

#endif // OPS_MLIR_RUNTIME_FUSION_PLANNER_H
