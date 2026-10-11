#include "ops-mlir/Conversion/Passes.h"
#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/DenseMap.h"

namespace ops_mlir::ops {

#define GEN_PASS_DEF_CONVERTOPSTOSCF
#include "ops-mlir/Conversion/Passes.h.inc"

namespace {

using namespace mlir;

/// A dat accessor of the loop being lowered.
struct Accessor {
  unsigned slot;                   // distinct dat (index into the function's memref arguments)
  Value memref;                    // the function argument
  SmallVector<int64_t> strides;    // stencil strides, OPS axis order
  int64_t ncomp;
};

/// Per reduction of the loop: the per-point accumulator and how to combine.
struct Accumulator {
  Value memref;
  ReductionKind kind;
  Type type;
};

Value neutralElement(OpBuilder &b, Location loc, ReductionKind kind, Type type) {
  if (auto ft = dyn_cast<FloatType>(type)) {
    const llvm::fltSemantics &sem = ft.getFloatSemantics();
    APFloat v(sem);
    switch (kind) {
    case ReductionKind::Add: v = APFloat::getZero(sem); break;
    case ReductionKind::Min: v = APFloat::getInf(sem); break;
    case ReductionKind::Max: v = APFloat::getInf(sem, /*Negative=*/true); break;
    }
    return arith::ConstantOp::create(b, loc, FloatAttr::get(type, v));
  }
  unsigned w = type.getIntOrFloatBitWidth();
  APInt v(w, 0);
  if (kind == ReductionKind::Min) v = APInt::getSignedMaxValue(w);
  if (kind == ReductionKind::Max) v = APInt::getSignedMinValue(w);
  return arith::ConstantOp::create(b, loc, IntegerAttr::get(type, v));
}

/// acc (op) v, with the C semantics of OPS_MIN / OPS_MAX (`a < b ? a : b`).
Value combine(OpBuilder &b, Location loc, ReductionKind kind, Value acc, Value v) {
  bool isFloat = isa<FloatType>(acc.getType());
  switch (kind) {
  case ReductionKind::Add:
    return isFloat ? Value(arith::AddFOp::create(b, loc, acc, v)) : Value(arith::AddIOp::create(b, loc, acc, v));
  case ReductionKind::Min:
  case ReductionKind::Max: {
    bool min = kind == ReductionKind::Min;
    Value cond = isFloat ? Value(arith::CmpFOp::create(b, loc, min ? arith::CmpFPredicate::OLT : arith::CmpFPredicate::OGT, acc, v))
                         : Value(arith::CmpIOp::create(b, loc, min ? arith::CmpIPredicate::slt : arith::CmpIPredicate::sgt, acc, v));
    return arith::SelectOp::create(b, loc, cond, acc, v);
  }
  }
  llvm_unreachable("unknown reduction kind");
}

MemRefType datMemRefType(DatType dat, bool soa) {
  MLIRContext *ctx = dat.getContext();
  int64_t rank = dat.getRank(), ncomp = dat.getNcomp();
  SmallVector<int64_t> shape(rank, ShapedType::kDynamic), strides(rank, ShapedType::kDynamic);
  // The last memref dimension of the block is x, the fastest: unit stride, or ncomp when the components of a point
  // are adjacent.
  strides[rank - 1] = (ncomp > 1 && !soa) ? ncomp : 1;
  if (ncomp > 1) {
    shape.push_back(ncomp);
    strides.push_back(soa ? ShapedType::kDynamic : 1);
  }
  return MemRefType::get(shape, dat.getElementType(), StridedLayoutAttr::get(ctx, ShapedType::kDynamic, strides));
}

struct ConvertOpsToSCFPass : public impl::ConvertOpsToSCFBase<ConvertOpsToSCFPass> {
  using ConvertOpsToSCFBase::ConvertOpsToSCFBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<ChainOp> chains(module.getOps<ChainOp>());
    for (ChainOp chain : chains)
      if (failed(lowerChain(chain, module))) return signalPassFailure();
  }

  LogicalResult lowerChain(ChainOp chain, ModuleOp module) {
    SmallVector<ParLoopOp> loops;
    bool hasCalls = false;
    for (Operation &op : chain.getBody().front()) {
      if (auto loop = dyn_cast<ParLoopOp>(&op)) loops.push_back(loop);
      if (isa<CallOp>(&op)) hasCalls = true;
    }
    if (loops.empty() || hasCalls) return success();
    if (loops.size() > 1)
      return chain.emitError("a group must have a single ops.par_loop (found ") << loops.size()
                                                                              << "): run ops-outline-groups first";
    bool called = false;
    module.walk([&](CallOp call) {
      if (call.getCallee() == chain.getSymName()) called = true;
    });
    if (called)
      return chain.emitError("the group is called from another chain of this module and its signature changes: split "
                             "the groups into modules of their own first");

    ParLoopOp loop = loops.front();
    MLIRContext *ctx = chain.getContext();
    Location loc = chain.getLoc();
    unsigned rank = loop.getRank();
    Block &chainBlock = chain.getBody().front();

    // --- ABI --------------------------------------------------------------------------------------------------
    auto chainArg = [&](Value v, StringRef what) -> BlockArgument {
      auto arg = dyn_cast<BlockArgument>(v);
      if (!arg || arg.getOwner() != &chainBlock) {
        loop.emitError("a ") << what << " operand of the loop is not an argument of the chain";
        return nullptr;
      }
      return arg;
    };
    SmallVector<BlockArgument> datArgs, scalarArgs;
    DenseMap<Value, unsigned> datSlot, scalarSlot;
    for (Value dat : loop.getDats()) {
      BlockArgument arg = chainArg(dat, "dat");
      if (!arg) return failure();
      if (datSlot.try_emplace(arg, datArgs.size()).second) datArgs.push_back(arg);
    }
    for (Value s : loop.getScalars()) {
      BlockArgument arg = chainArg(s, "scalar");
      if (!arg) return failure();
      if (scalarSlot.try_emplace(arg, scalarArgs.size()).second) scalarArgs.push_back(arg);
    }
    unsigned numDats = datArgs.size();

    SmallVector<Type> inputs;
    for (BlockArgument d : datArgs) inputs.push_back(datMemRefType(cast<DatType>(d.getType()), soa));
    Type indexType = IndexType::get(ctx);
    for (unsigned i = 0; i < numDats * rank + 2 * rank + rank; ++i) inputs.push_back(indexType);
    for (BlockArgument s : scalarArgs) inputs.push_back(s.getType());
    SmallVector<Type> results(loop->getResultTypes());
    const bool gpu = target == "gpu";
    if (target != "cpu" && !gpu) return chain.emitError("option 'target' must be cpu or gpu, got '") << target << "'";
    if (gpu) {
      // The results come back through device buffers: one rank-0 memref per reduction, after the scalars.
      for (Type t : results) inputs.push_back(MemRefType::get({}, t));
      results.clear();
    }

    unsigned shiftBase = numDats, rangeBase = shiftBase + numDats * rank, dispBase = rangeBase + 2 * rank,
             scalarBase = dispBase + rank, bufferBase = scalarBase + scalarArgs.size();

    // The function is named by `ops.func_name` (the runtime picks group_<hash>) or the chain; a declaration of that
    // name, as a caller in the same module needs, is replaced by the definition.
    StringRef name = chain.getSymName();
    if (auto given = chain->getAttrOfType<StringAttr>("ops.func_name")) name = given.getValue();
    if (auto existing = module.lookupSymbol<func::FuncOp>(name)) {
      if (!existing.isExternal()) return chain.emitError("a function named @") << name << " already exists";
      existing.erase();
    }
    OpBuilder b(chain);
    auto func = func::FuncOp::create(b, loc, name, FunctionType::get(ctx, inputs, results));
    func.setPublic();
    func->setAttr("llvm.emit_c_interface", UnitAttr::get(ctx));
    for (unsigned i = 0; i < numDats; ++i) func.setArgAttr(i, "llvm.noalias", UnitAttr::get(ctx));
    Builder bb(ctx);
    // Position of the aligned pointer of each dat once the memref arguments are expanded to descriptors (allocated
    // pointer, aligned pointer, offset, sizes, strides).
    SmallVector<int64_t> noaliasArgs;
    for (unsigned pos = 0; Type t : inputs) {
      auto memref = dyn_cast<MemRefType>(t);
      if (!memref) break;
      noaliasArgs.push_back(pos + 1);
      pos += 3 + 2 * memref.getRank();
    }
    func->setAttr("ops.abi", bb.getDictionaryAttr({bb.getNamedAttr("noalias_args", bb.getDenseI64ArrayAttr(noaliasArgs)),
                                                   bb.getNamedAttr("rank", bb.getI64IntegerAttr(rank)),
                                                   bb.getNamedAttr("num_dats", bb.getI64IntegerAttr(numDats)),
                                                   bb.getNamedAttr("num_scalars", bb.getI64IntegerAttr(scalarArgs.size())),
                                                   bb.getNamedAttr("num_reductions", bb.getI64IntegerAttr(loop->getNumResults())),
                                                   bb.getNamedAttr("gpu", bb.getBoolAttr(gpu)),
                                                   bb.getNamedAttr("soa", bb.getBoolAttr(soa))}));
    Block *entry = func.addEntryBlock();
    b.setInsertionPointToEnd(entry);

    // --- chain-level ops other than the loop (constants...) -------------------------------------------------------
    IRMapping mapper;
    for (auto [i, s] : llvm::enumerate(scalarArgs)) mapper.map(s, entry->getArgument(scalarBase + i));
    for (Operation &op : chainBlock) {
      if (isa<ParLoopOp, ReturnOp>(&op)) continue;
      b.clone(op, mapper);
    }

    // --- the loop ---------------------------------------------------------------------------------------------
    // A range operand that is a constant (ops-specialize) is used as such; otherwise it is the function argument.
    auto rangeValue = [&](unsigned slot) -> Value {
      Value operand = loop.getRange()[slot];
      if (operand.getDefiningOp() && matchPattern(operand, m_Constant())) return mapper.lookup(operand);
      return entry->getArgument(rangeBase + slot);
    };
    SmallVector<ReductionKind> kinds;
    for (unsigned r = 0; r < loop->getNumResults(); ++r) kinds.push_back(loop.getReductionKind(r));
    Value one = arith::ConstantIndexOp::create(b, loc, 1);

    // The shell around the body of one point: where the body goes (`body`), the coordinate of the point in each
    // dimension (`iv`), and what to look through to find the ops to rewrite (`walkRoot`).
    Block *body = nullptr;
    Operation *walkRoot = nullptr;
    scf::ParallelOp parallel;
    Value innerIv;
    SmallVector<Value, 3> gpuIvs;
    Operation *gpuActive = nullptr;     // the scf.if that masks the threads outside the range
    gpu::LaunchOp gpuLaunch;
    SmallVector<int64_t, 3> gpuBlock;
    const bool innerLoop = !gpu && rank > 1;
    auto iv = [&](unsigned d) -> Value {
      if (gpu) return gpuIvs[d];
      if (innerLoop && d == 0) return innerIv;
      return body->getArgument(rank - 1 - d);
    };
    if (!gpu) {
      // The outer dimensions (z, y) run in parallel; x, the contiguous one, is a sequential loop inside, which the
      // compiler vectorises. (With one dimension only, that one is the parallel loop.)
      SmallVector<Value> lbs, ubs, steps, inits;
      for (int d = rank - 1; d >= (innerLoop ? 1 : 0); --d) { // outermost (z) first
        lbs.push_back(rangeValue(2 * d));
        ubs.push_back(rangeValue(2 * d + 1));
        steps.push_back(one);
      }
      for (unsigned r = 0; r < loop->getNumResults(); ++r) inits.push_back(neutralElement(b, loc, kinds[r], loop->getResult(r).getType()));
      parallel = scf::ParallelOp::create(b, loc, lbs, ubs, steps, inits);
      body = parallel.getBody();
      walkRoot = parallel;
      b.setInsertionPointToStart(body);
    } else {
      // Thread block and grid: x is the fastest dimension of memory and of the block.
      SmallVector<int64_t, 3> bs;
      for (StringRef part : llvm::split(block, ',')) {
        int64_t v;
        if (part.trim().getAsInteger(10, v) || v < 1) return chain.emitError("option 'block' must be positive integers x,y,z, got '") << block << "'";
        bs.push_back(v);
      }
      bs.resize(3, 1);
      for (unsigned d = rank; d < 3; ++d)
        if (bs[d] != 1) return chain.emitError("the block size must be 1 in the dimensions the block does not have (rank ") << rank << ", block " << block << ")";
      Value zero = arith::ConstantIndexOp::create(b, loc, 0);
      SmallVector<Value, 3> extents, grid(3, one), blockSize;
      Value nonEmpty;
      for (unsigned d = 0; d < 3; ++d) blockSize.push_back(arith::ConstantIndexOp::create(b, loc, bs[d]));
      for (unsigned d = 0; d < rank; ++d) {
        Value extent = arith::SubIOp::create(b, loc, rangeValue(2 * d + 1), rangeValue(2 * d));
        Value positive = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt, extent, zero);
        nonEmpty = nonEmpty ? Value(arith::AndIOp::create(b, loc, nonEmpty, positive)) : positive;
        grid[d] = arith::CeilDivSIOp::create(b, loc, arith::MaxSIOp::create(b, loc, extent, zero), blockSize[d]);
      }
      auto nonEmptyIf = scf::IfOp::create(b, loc, nonEmpty, /*withElseRegion=*/false);
      b.setInsertionPointToStart(&nonEmptyIf.getThenRegion().front());
      auto launch = gpu::LaunchOp::create(b, loc, grid[0], grid[1], grid[2], blockSize[0], blockSize[1], blockSize[2]);
      b.setInsertionPointToStart(&launch.getBody().front());
      gpu::TerminatorOp::create(b, loc);
      b.setInsertionPoint(launch.getBody().front().getTerminator());
      Value inRange;
      auto blockIds = launch.getBlockIds();
      auto threadIds = launch.getThreadIds();
      Value ids[3][2] = {{blockIds.x, threadIds.x}, {blockIds.y, threadIds.y}, {blockIds.z, threadIds.z}};
      for (unsigned d = 0; d < rank; ++d) {
        Value base = arith::MulIOp::create(b, loc, ids[d][0], blockSize[d]);
        Value local = arith::AddIOp::create(b, loc, base, ids[d][1]);
        Value index = arith::AddIOp::create(b, loc, local, rangeValue(2 * d));
        gpuIvs.push_back(index);
        Value inside = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, index, rangeValue(2 * d + 1));
        inRange = inRange ? Value(arith::AndIOp::create(b, loc, inRange, inside)) : inside;
      }
      auto active = scf::IfOp::create(b, loc, inRange, /*withElseRegion=*/false);
      body = &active.getThenRegion().front();
      walkRoot = launch;
      gpuActive = active;
      gpuLaunch = launch;
      gpuBlock.assign(bs.begin(), bs.end());
      b.setInsertionPointToStart(body);
    }

    // Accessors, scalars and reductions of the loop body, as values the cloned body can use until it is rewritten.
    SmallVector<Accessor> accessors;
    for (unsigned i = 0; i < loop.getNumDats(); ++i) {
      unsigned slot = datSlot.lookup(loop.getDats()[i]);
      StencilAttr stencil = loop.getStencil(i);
      Accessor a;
      a.slot = slot;
      a.memref = entry->getArgument(slot);
      a.strides.assign(stencil.getStrides().begin(), stencil.getStrides().end());
      a.ncomp = cast<DatType>(loop.getDats()[i].getType()).getNcomp();
      accessors.push_back(a);
      auto cast = UnrealizedConversionCastOp::create(b, loc, loop.getAccessorArg(i).getType(), a.memref);
      cast->setAttr("ops.accessor", bb.getI64IntegerAttr(i));
      mapper.map(loop.getAccessorArg(i), cast.getResult(0));
    }
    for (unsigned i = 0; i < loop.getScalars().size(); ++i)
      mapper.map(loop.getScalarArg(i), entry->getArgument(scalarBase + scalarSlot.lookup(loop.getScalars()[i])));
    SmallVector<Accumulator> accumulators;
    for (unsigned r = 0; r < loop->getNumResults(); ++r) {
      Type t = loop->getResult(r).getType();
      // On the GPU every thread of a block takes part in the reduction after the body: the accumulators live outside the
      // mask of the threads that are in the range.
      OpBuilder::InsertionGuard accumulatorGuard(b);
      if (gpu) b.setInsertionPoint(gpuActive);
      auto allocaOp = memref::AllocaOp::create(b, loc, MemRefType::get({}, t));
      // Re-initialised per iteration of the parallel loop: ops-hoist-accumulators moves it out of the loop.
      allocaOp->setAttr("ops.accumulator", UnitAttr::get(ctx));
      Value alloca = allocaOp;
      memref::StoreOp::create(b, loc, neutralElement(b, loc, kinds[r], t), alloca, ValueRange{});
      accumulators.push_back({alloca, kinds[r], t});
      auto cast = UnrealizedConversionCastOp::create(b, loc, loop.getReductionArg(r).getType(), alloca);
      cast->setAttr("ops.reduction", bb.getI64IntegerAttr(r));
      mapper.map(loop.getReductionArg(r), cast.getResult(0));
    }

    if (innerLoop) {
      auto inner = scf::ForOp::create(b, loc, rangeValue(0), rangeValue(1), one);
      innerIv = inner.getInductionVar();
      b.setInsertionPointToStart(inner.getBody());
    }
    Operation *firstCloned = nullptr;
    for (Operation &op : loop.getBodyBlock().without_terminator()) {
      Operation *c = b.clone(op, mapper);
      if (!firstCloned) firstCloned = c;
    }

    // Rewrite the ops of the dialect that the clones contain, nested ones included.
    SmallVector<Operation *> toRewrite;
    walkRoot->walk([&](Operation *op) {
      if (isa<ReadOp, ReadDynOp, WriteOp, IndexOp, ReduceOp, GblReadOp, ConstRefOp>(op)) toRewrite.push_back(op);
    });

    auto accessorOf = [&](Value acc) -> const Accessor * {
      auto cast = acc.getDefiningOp<UnrealizedConversionCastOp>();
      if (!cast) return nullptr;
      auto idx = cast->getAttrOfType<IntegerAttr>("ops.accessor");
      return idx ? &accessors[idx.getInt()] : nullptr;
    };

    for (Operation *op : toRewrite) {
      OpBuilder rb(op);
      Location l = op->getLoc();
      // Memref indices of an access at constant offsets (dynamic offsets: `dyn`), memory order.
      auto indices = [&](const Accessor &a, ArrayRef<int64_t> offsets, ValueRange dyn, int64_t comp) {
        SmallVector<Value> idx(rank + (a.ncomp > 1 ? 1 : 0));
        for (unsigned d = 0; d < rank; ++d) {
          Value v = entry->getArgument(shiftBase + a.slot * rank + d);
          if (a.strides[d] != 0) v = arith::AddIOp::create(rb, l, v, iv(d));
          if (!offsets.empty() && offsets[d] != 0) v = arith::AddIOp::create(rb, l, v, arith::ConstantIndexOp::create(rb, l, offsets[d]));
          if (!dyn.empty()) {
            Value o = dyn[d];
            if (!o.getType().isIndex()) o = arith::IndexCastOp::create(rb, l, rb.getIndexType(), o);
            v = arith::AddIOp::create(rb, l, v, o);
          }
          idx[rank - 1 - d] = v;
        }
        if (a.ncomp > 1) idx[rank] = arith::ConstantIndexOp::create(rb, l, comp);
        return idx;
      };

      if (auto read = dyn_cast<ReadOp>(op)) {
        const Accessor *a = accessorOf(read.getAcc());
        if (!a) return read.emitError("access through a value that is not an accessor of the loop");
        auto load = memref::LoadOp::create(rb, l, a->memref, indices(*a, read.getOffsets(), {}, read.getComp()));
        read.getResult().replaceAllUsesWith(load);
      } else if (auto dyn = dyn_cast<ReadDynOp>(op)) {
        const Accessor *a = accessorOf(dyn.getAcc());
        if (!a) return dyn.emitError("access through a value that is not an accessor of the loop");
        auto load = memref::LoadOp::create(rb, l, a->memref, indices(*a, {}, dyn.getOffsets(), dyn.getComp()));
        dyn.getResult().replaceAllUsesWith(load);
      } else if (auto write = dyn_cast<WriteOp>(op)) {
        const Accessor *a = accessorOf(write.getAcc());
        if (!a) return write.emitError("access through a value that is not an accessor of the loop");
        memref::StoreOp::create(rb, l, write.getValue(), a->memref, indices(*a, write.getOffsets(), {}, write.getComp()));
      } else if (auto index = dyn_cast<IndexOp>(op)) {
        unsigned d = index.getDim();
        Value v = arith::AddIOp::create(rb, l, iv(d), entry->getArgument(dispBase + d));
        index.getResult().replaceAllUsesWith(arith::IndexCastOp::create(rb, l, rb.getI32Type(), v));
      } else if (auto reduce = dyn_cast<ReduceOp>(op)) {
        auto cast = reduce.getRed().getDefiningOp<UnrealizedConversionCastOp>();
        auto r = cast ? cast->getAttrOfType<IntegerAttr>("ops.reduction") : IntegerAttr();
        if (!r) return reduce.emitError("reduction through a value that is not a reduction of the loop");
        const Accumulator &acc = accumulators[r.getInt()];
        Value cur = memref::LoadOp::create(rb, l, acc.memref, ValueRange{});
        memref::StoreOp::create(rb, l, combine(rb, l, acc.kind, cur, reduce.getValue()), acc.memref, ValueRange{});
      } else {
        return op->emitError("must not remain in a loop body when lowering a group (materialise and specialise first)");
      }
      op->erase();
    }
    // The placeholders have no users left.
    {
      SmallVector<Operation *> dead;
      walkRoot->walk([&](UnrealizedConversionCastOp cast) {
        if (cast.use_empty() && (cast->hasAttr("ops.accessor") || cast->hasAttr("ops.reduction"))) dead.push_back(cast);
      });
      for (Operation *op : dead) op->erase();
    }

    // The partial reductions of the points.
    if (gpu) {
      // The threads of a warp combine their partial values with shuffles and lane 0 adds the result to the buffer of the
      // reduction with an atomic (one atomic per warp). A block that is not a whole number of warps has every thread do its own.
      b.setInsertionPoint(gpuLaunch.getBody().front().getTerminator());
      const int64_t threads = gpuBlock[0] * gpuBlock[1] * gpuBlock[2];
      const bool warps = threads % 32 == 0;
      Value lane0;
      if (warps) {
        auto tid = gpuLaunch.getThreadIds();
        Value c32 = arith::ConstantIndexOp::create(b, loc, 32);
        Value lin = arith::AddIOp::create(b, loc, tid.x,
                    arith::MulIOp::create(b, loc, arith::ConstantIndexOp::create(b, loc, gpuBlock[0]),
                    arith::AddIOp::create(b, loc, tid.y, arith::MulIOp::create(b, loc, arith::ConstantIndexOp::create(b, loc, gpuBlock[1]), tid.z))));
        Value lane = arith::RemUIOp::create(b, loc, lin, c32);
        lane0 = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lane, arith::ConstantIndexOp::create(b, loc, 0));
      }
      for (auto [i, acc] : llvm::enumerate(accumulators)) {
        Value partial = memref::LoadOp::create(b, loc, acc.memref, ValueRange{});
        if (warps) {
          Value width = arith::ConstantIntOp::create(b, loc, 32, 32);
          for (int offset = 16; offset >= 1; offset /= 2) {
            Value delta = arith::ConstantIntOp::create(b, loc, offset, 32);
            auto shuffled = gpu::ShuffleOp::create(b, loc, partial, delta, width, gpu::ShuffleMode::XOR);
            partial = combine(b, loc, acc.kind, partial, shuffled.getShuffleResult());
          }
        }
        OpBuilder::InsertionGuard guard(b);
        if (warps) {
          auto onlyLane0 = scf::IfOp::create(b, loc, lane0, /*withElseRegion=*/false);
          b.setInsertionPointToStart(&onlyLane0.getThenRegion().front());
        }
        bool isFloat = isa<FloatType>(acc.type);
        arith::AtomicRMWKind kind = acc.kind == ReductionKind::Add ? (isFloat ? arith::AtomicRMWKind::addf : arith::AtomicRMWKind::addi)
                                    : acc.kind == ReductionKind::Min ? (isFloat ? arith::AtomicRMWKind::minnumf : arith::AtomicRMWKind::mins)
                                                                    : (isFloat ? arith::AtomicRMWKind::maxnumf : arith::AtomicRMWKind::maxs);
        memref::AtomicRMWOp::create(b, loc, kind, partial, entry->getArgument(bufferBase + i), ValueRange{});
      }
    } else {
      b.setInsertionPointToEnd(body);
    }
    if (!gpu && !accumulators.empty()) {
      SmallVector<Value> partials;
      for (const Accumulator &acc : accumulators) partials.push_back(memref::LoadOp::create(b, loc, acc.memref, ValueRange{}));
      auto reduce = scf::ReduceOp::create(b, loc, partials);
      for (auto [i, acc] : llvm::enumerate(accumulators)) {
        Block &block = reduce.getReductions()[i].front();
        b.setInsertionPointToEnd(&block);
        Value v = combine(b, loc, acc.kind, block.getArgument(0), block.getArgument(1));
        scf::ReduceReturnOp::create(b, loc, v);
      }
    }

    // --- return ---------------------------------------------------------------------------------------------------
    b.setInsertionPointToEnd(entry);
    SmallVector<Value> returned;
    auto ret = cast<ReturnOp>(chainBlock.getTerminator());
    for (Value v : ret.getOperands()) {
      auto res = dyn_cast<OpResult>(v);
      if (!res || res.getOwner() != loop.getOperation())
        return ret.emitError("a returned value is not a reduction of the loop");
      if (!gpu) returned.push_back(parallel.getResult(res.getResultNumber()));
    }
    func::ReturnOp::create(b, loc, returned);

    chain.erase();
    return success();
  }
};

} // namespace

} // namespace ops_mlir::ops
