#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"
#include "ops-mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"

#include <map>
#include <tuple>

namespace ops_mlir::ops {

#define GEN_PASS_DEF_FORWARDSTORES
#include "ops-mlir/Transforms/Passes.h.inc"

namespace {

using namespace mlir;

/// (accessor, offsets, component) of a point of a dat.
using Slot = std::tuple<const void *, std::vector<int64_t>, int64_t>;

struct FuseStoresPass : public impl::ForwardStoresBase<FuseStoresPass> {
  void runOnOperation() override {
    getOperation().walk([&](ParLoopOp loop) { processLoop(loop); });
  }

  void processLoop(ParLoopOp loop) {
    // An accessor can be forwarded only if no other accessor of the loop is for the same dat.
    DenseMap<Value, unsigned> uses;
    for (Value dat : loop.getDats()) ++uses[dat];
    auto exclusive = [&](Value acc) {
      auto arg = dyn_cast<BlockArgument>(acc);
      if (!arg || arg.getOwner() != &loop.getBodyBlock() || arg.getArgNumber() >= loop.getNumDats()) return false;
      return uses.lookup(loop.getDats()[arg.getArgNumber()]) == 1;
    };

    std::map<Slot, Value> stored;
    SmallVector<Operation *> dead;
    for (Operation &op : loop.getBodyBlock().without_terminator()) {
      if (auto write = dyn_cast<WriteOp>(&op)) {
        if (exclusive(write.getAcc()))
          stored[{write.getAcc().getAsOpaquePointer(), std::vector<int64_t>(write.getOffsets().begin(), write.getOffsets().end()),
                  static_cast<int64_t>(write.getComp())}] = write.getValue();
        continue;
      }
      if (auto read = dyn_cast<ReadOp>(&op)) {
        auto it = stored.find({read.getAcc().getAsOpaquePointer(), std::vector<int64_t>(read.getOffsets().begin(), read.getOffsets().end()),
                               static_cast<int64_t>(read.getComp())});
        if (it != stored.end() && it->second.getType() == read.getType()) {
          read.getResult().replaceAllUsesWith(it->second);
          dead.push_back(read);
        }
        continue;
      }
      if (op.getNumRegions() > 0) {
        // A nested region may write anything it touches: forget those accessors.
        op.walk([&](WriteOp w) {
          for (auto it = stored.begin(); it != stored.end();)
            it = std::get<0>(it->first) == w.getAcc().getAsOpaquePointer() ? stored.erase(it) : std::next(it);
        });
      }
    }
    for (Operation *op : dead) op->erase();

    // Narrow the access modes to what the body does.
    DenseMap<Value, std::pair<bool, bool>> usage; // accessor -> (read, written)
    loop.getBody().walk([&](Operation *op) {
      if (auto r = dyn_cast<ReadOp>(op)) usage[r.getAcc()].first = true;
      else if (auto r = dyn_cast<ReadDynOp>(op)) usage[r.getAcc()].first = true;
      else if (auto w = dyn_cast<WriteOp>(op)) usage[w.getAcc()].second = true;
    });
    SmallVector<Attribute> accesses(loop.getAccesses().begin(), loop.getAccesses().end());
    bool changed = false;
    for (unsigned i = 0; i < loop.getNumDats(); ++i) {
      if (!exclusive(loop.getAccessorArg(i))) continue;
      auto [read, written] = usage.lookup(loop.getAccessorArg(i));
      AccessMode mode = loop.getAccess(i);
      AccessMode narrowed = mode;
      if (mode == AccessMode::RW && !read && written) narrowed = AccessMode::Write;
      else if (mode == AccessMode::RW && read && !written) narrowed = AccessMode::Read;
      if (narrowed != mode) {
        accesses[i] = AccessModeAttr::get(loop.getContext(), narrowed);
        changed = true;
      }
    }
    if (changed) loop.setAccessesAttr(ArrayAttr::get(loop.getContext(), accesses));
  }
};

} // namespace

} // namespace ops_mlir::ops
