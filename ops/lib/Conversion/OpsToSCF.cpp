#include "ops-mlir/Conversion/Passes.h"
#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

    unsigned shiftBase = numDats, rangeBase = shiftBase + numDats * rank, dispBase = rangeBase + 2 * rank,
             scalarBase = dispBase + rank;

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
                                                   bb.getNamedAttr("num_reductions", bb.getI64IntegerAttr(results.size())),
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
    SmallVector<Value> lbs, ubs, steps, inits;
    Value one = arith::ConstantIndexOp::create(b, loc, 1);
    // The outer dimensions (z, y) run in parallel; x, the contiguous one, is a sequential loop inside, which the
    // compiler vectorises. (With one dimension only, that one is the parallel loop.)
    const bool innerLoop = rank > 1;
    for (int d = rank - 1; d >= (innerLoop ? 1 : 0); --d) { // outermost (z) first
      lbs.push_back(rangeValue(2 * d));
      ubs.push_back(rangeValue(2 * d + 1));
      steps.push_back(one);
    }
    SmallVector<ReductionKind> kinds;
    for (unsigned r = 0; r < loop->getNumResults(); ++r) {
      kinds.push_back(loop.getReductionKind(r));
      inits.push_back(neutralElement(b, loc, kinds[r], loop->getResult(r).getType()));
    }
    auto parallel = scf::ParallelOp::create(b, loc, lbs, ubs, steps, inits);
    Block *body = parallel.getBody();
    b.setInsertionPointToStart(body);

    Value innerIv;
    auto iv = [&](unsigned d) -> Value {
      if (innerLoop && d == 0) return innerIv;
      return body->getArgument(rank - 1 - d);
    };

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
    parallel.getBody()->walk([&](Operation *op) {
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
    for (Operation &op : llvm::make_early_inc_range(*body))
      if (isa<UnrealizedConversionCastOp>(op) && op.use_empty()) op.erase();

    // The partial reductions of the points.
    b.setInsertionPointToEnd(body);
    if (!accumulators.empty()) {
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
      returned.push_back(parallel.getResult(res.getResultNumber()));
    }
    func::ReturnOp::create(b, loc, returned);

    chain.erase();
    return success();
  }
};

} // namespace

} // namespace ops_mlir::ops
