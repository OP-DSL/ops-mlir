//===- Executor.h - the chain executor of the OPS JIT -----------------------*- C++ -*-===//
//
// Plan section 4.5. Registered with OPS by ops_jit_init(); at every flush of the lazy queue it receives the loops in
// program order. Runs of claimable loops are lifted into an ops.chain, planned (fused), lowered and compiled per
// group, and executed with the per-loop steps of ops_lazy_executor.h around each group. Everything else runs with
// ops_lazy_execute_default().
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_RUNTIME_EXECUTOR_H
#define OPS_MLIR_RUNTIME_EXECUTOR_H

#include "ops-mlir/Runtime/Artifact.h"
#include "ops-mlir/Runtime/Binding.h"
#include "ops-mlir/Runtime/GroupCompiler.h"
#include "ops-mlir/Runtime/Options.h"

#include "opdsl/Cache/PersistentCache.h"
#include "opdsl/JIT/JITSession.h"
#include "opdsl/Pipeline/PipelineRegistry.h"

#include "mlir/IR/MLIRContext.h"

#include "ops_lib_core.h"
#include "ops_internal2.h"

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ops_mlir::runtime {

/// What the executor does with a run of loops of one structure.
struct Plan {
  struct Group {
    GroupRecipe recipe;
    std::shared_ptr<GroupCode> code;
  };
  std::vector<Group> groups;
  std::vector<ReductionRef> reductions;
  std::string declined; ///< non-empty: the run cannot be planned (reason)
};

struct Stats {
  uint64_t flushes = 0, loopsSeen = 0, loopsClaimed = 0, loopsDeclined = 0, groupsExecuted = 0;
  uint64_t planBuilds = 0, planHits = 0, planFailures = 0;
  uint64_t verifyGroups = 0, verifyMismatches = 0;
  std::map<std::string, uint64_t> declineReasons;
  double tClaim = 0, tPlan = 0, tLift = 0, tFrontend = 0, tPlanPasses = 0, tExecute = 0, tBind = 0;
};

class Executor {
public:
  Executor();

  /// The registered chain executor function (OPS calls this).
  static void run(OPS_instance *instance, ops_kernel_descriptor **descs, int n, void *ctx);
  void execute(OPS_instance *instance, ops_kernel_descriptor **descs, int n);

  void printStats() const;
  Stats &stats() { return stats_; }
  const CompileStats &compileStats() const { return compiler_->stats(); }

  /// Whether this executor may run loops that are on a device (set by device builds).
  bool deviceSupported = false;

private:
  struct EffectiveRun;
  /// Runs descs[0..n) (all claimed) with the JIT. False: could not (the caller runs them with OPS).
  bool runJit(OPS_instance *instance, ops_kernel_descriptor **descs, int n, const std::vector<const KernelArtifact *> &artifacts);
  std::shared_ptr<Plan> buildPlan(ops_kernel_descriptor **run, int n, const std::vector<std::vector<int>> &ranges,
                                  const std::vector<const KernelArtifact *> &artifacts, const std::string &planKey);
  std::string structuralKey(ops_kernel_descriptor **run, int n, const std::vector<std::vector<int>> &ranges,
                            const std::vector<const KernelArtifact *> &artifacts, bool soa);
  /// OPS_JIT_VERIFY: the group has just run; compare it with the stock OPS code on the same inputs (see Executor.cpp).
  struct VerifyState;

  mlir::DialectRegistry dialects_;
  std::unique_ptr<mlir::MLIRContext> ctx_;
  std::unique_ptr<ArtifactCache> artifacts_;
  opdsl::PipelineRegistry registry_;
  std::unique_ptr<GroupCompiler> compiler_;
  std::unordered_map<std::string, std::shared_ptr<Plan>> plans_;
  Stats stats_;
};

/// The process-wide executor.
Executor &executor();

} // namespace ops_mlir::runtime

#endif // OPS_MLIR_RUNTIME_EXECUTOR_H
