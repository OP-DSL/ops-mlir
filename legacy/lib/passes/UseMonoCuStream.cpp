//===- UseMonoCuStream.cpp - Force kernels onto a single CUDA stream ----===//
//
// Part of OPS-MLIR Project
// Author: Prakanth Thilakaraj
// Date: August 2026
//
// This file is distributed under the MIT License.
// See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

#include "passes/UseMonoCuStream.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

namespace ops_mlir {

llvm::Error useMonoCudaStream(llvm::Module *llvmModule) {
  llvm::Function *streamCreateFn = llvmModule->getFunction("mgpuStreamCreate");
  llvm::Function *streamDestroyFn = llvmModule->getFunction("mgpuStreamDestroy");
  if (!streamCreateFn && !streamDestroyFn)
    return llvm::Error::success();

  llvm::Function *persistentStreamFn = llvm::cast<llvm::Function>(
      llvmModule
          ->getOrInsertFunction(
              kPersistentCudaStreamGetterName,
              llvm::FunctionType::get(
                  llvm::PointerType::getUnqual(llvmModule->getContext()), {}))
          .getCallee());

  if (streamCreateFn) {
    llvm::SmallVector<llvm::CallInst *> calls;
    for (llvm::User *user : streamCreateFn->users())
      if (auto *call = llvm::dyn_cast<llvm::CallInst>(user))
        calls.push_back(call);
    for (llvm::CallInst *call : calls) {
      llvm::CallInst *replacement =
          llvm::CallInst::Create(persistentStreamFn, {}, "", call->getIterator());
      call->replaceAllUsesWith(replacement);
      call->eraseFromParent();
    }
  }

  if (streamDestroyFn) {
    llvm::SmallVector<llvm::CallInst *> calls;
    for (llvm::User *user : streamDestroyFn->users())
      if (auto *call = llvm::dyn_cast<llvm::CallInst>(user))
        calls.push_back(call);
    for (llvm::CallInst *call : calls)
      call->eraseFromParent();
  }

  return llvm::Error::success();
}

} // namespace ops_mlir
