#include "ops-mlir/Pipelines/GroupModules.h"

#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/Support/raw_ostream.h"

namespace ops_mlir::ops {

using namespace mlir;

std::vector<GroupModule> extractGroupModules(ModuleOp planned) {
  std::vector<GroupModule> out;
  for (ChainOp chain : planned.getOps<ChainOp>()) {
    for (CallOp call : chain.getBody().getOps<CallOp>()) {
      auto group = planned.lookupSymbol<ChainOp>(call.getCallee());
      if (!group) {
        call.emitError("the call does not name an ops.chain of the module");
        return {};
      }
      GroupModule g;
      g.name = group.getSymName().str();
      g.module = ModuleOp::create(group.getLoc());
      OpBuilder b(g.module->getBody(), g.module->getBody()->begin());
      b.clone(*group);
      out.push_back(std::move(g));
    }
  }
  return out;
}

std::string moduleBytecode(ModuleOp module) {
  std::string bytes;
  llvm::raw_string_ostream os(bytes);
  BytecodeWriterConfig config;
  if (failed(writeBytecodeToFile(module, os, config))) return {};
  return bytes;
}

void registerJITDialects(DialectRegistry &registry) {
  registry.insert<OPSDialect, arith::ArithDialect, func::FuncDialect, math::MathDialect, memref::MemRefDialect,
                  scf::SCFDialect, cf::ControlFlowDialect, LLVM::LLVMDialect, omp::OpenMPDialect>();
}

} // namespace ops_mlir::ops
