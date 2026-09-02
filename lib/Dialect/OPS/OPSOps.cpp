//===- OPSOps.cpp - OPS Dialect Operation Definitions --------*- C++ -*-===//
//
// Part of OPS-MLIR Project
// Author: Prakanth Thilakaraj
// Date: June 2026
//
// This file is distributed under the MIT License.
// See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

#include "Dialect/OPS/OPSOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace ops_mlir::ops;

namespace ops_mlir::ops {

void ParLoopOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

ParseResult ParLoopOp::parse(OpAsmParser &parser, OperationState &result) {
  if (failed(parser.parseOptionalAttrDict(result.attributes))) {
    return failure();
  }
  return success();
}

} // namespace ops_mlir::ops
