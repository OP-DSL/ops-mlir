//===- ops-opt.cpp - mlir-opt with the ops dialect and its passes --------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "ops-mlir/Analysis/Passes.h"
#include "ops-mlir/Dialect/OPS/IR/OPSDialect.h"
#include "ops-mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  ops_mlir::ops::registerAnalysisPasses();
  ops_mlir::ops::registerOPSTransformsPasses();

  mlir::DialectRegistry registry;
  registry.insert<ops_mlir::ops::OPSDialect, mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  mlir::registerAllExtensions(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "OPS MLIR optimizer driver\n", registry));
}
