//===- Binding.h - from queued loops to the arguments of a compiled group ----*- C++ -*-===//
//
// Plan sections 4.5 (binding recipe) and 4.6 (ABI of a compiled group). A compiled group is a function
//
//   (dat memref x D, shift x (D * rank), range lo/hi x (2 * rank), disp x rank, scalar x S) -> reduction values
//
// called through the packed wrapper of opdsl::JITSession. The recipe says, for each of those, where the value comes
// from in the current run of descriptors (dats by slot, ranges by loop, globals and constants by name); binding
// evaluates it.
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_RUNTIME_BINDING_H
#define OPS_MLIR_RUNTIME_BINDING_H

#include "ops-mlir/Runtime/Lifter.h"
#include "ops-mlir/Pipelines/GroupModules.h"

#include "opdsl/JIT/JITSession.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ops_mlir::runtime {

/// How to call one group of a planned chain.
struct GroupRecipe {
  int rank = 0;
  std::vector<int> datSlots;          ///< memref arguments: dats by slot of the run
  std::vector<ChainArg> range;        ///< 2 * rank entries: lo0, hi0, lo1, hi1, ...
  std::vector<ChainArg> scalars;      ///< scalar arguments (globals and constants)
  std::vector<int> reductionOf;       ///< per result of the group: index into Lifted::reductions, or -1
  std::vector<std::string> resultElem; ///< element type name per result
  std::vector<int> members;           ///< loops of the run the group executes, in order
};

/// The recipes of the groups of a planned module, in call order (same order as extractGroupModules). `lifted` supplies
/// the meaning of the chain arguments.
bool buildRecipes(const Lifted &lifted, mlir::ModuleOp planned, std::vector<GroupRecipe> &recipes, std::string &error);

/// Storage for one call: build it, then call().
class Invocation {
public:
  /// Evaluates `recipe` for the loops `run` (with their local `ranges`). `soa`: OPS stores components apart.
  bool bind(const GroupRecipe &recipe, ops_kernel_descriptor *const *run, const std::vector<std::vector<int>> &ranges,
            const std::vector<ops_dat> &slots, bool soa, std::string &error);

  /// Calls the compiled group. After it, result<T>(i) reads the reduction values.
  void call(opdsl::JITSession::PackedFn fn);

  /// Result `i` (in the order of the group's results) as a double, int64 or float according to `elem`.
  double resultAsDouble(size_t i) const;
  int64_t resultAsInt(size_t i) const;
  const char *resultPointer(size_t i) const { return results.data() + resultOffset[i]; }

private:
  struct Memref {
    std::array<int64_t, 3 + 2 * 4> words{}; // allocated, aligned, offset, sizes[r], strides[r]
  };
  struct Scalar {
    alignas(8) char bytes[8] = {0};
  };
  std::vector<Memref> memrefs;
  std::vector<Memref *> memrefPointers;
  std::vector<int64_t> indexArgs; ///< shifts, ranges, displacements
  std::vector<Scalar> scalarArgs;
  std::vector<void *> argv;
  std::vector<char> results;
  std::vector<size_t> resultOffset;
  std::vector<std::string> resultElem;
};

/// Natural size and alignment of an element type name.
size_t sizeOfElem(const std::string &elem);

} // namespace ops_mlir::runtime

#endif // OPS_MLIR_RUNTIME_BINDING_H
