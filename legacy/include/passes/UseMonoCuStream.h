#ifndef OPS_MLIR_PASSES_USE_MONO_CU_STREAM_H
#define OPS_MLIR_PASSES_USE_MONO_CU_STREAM_H

namespace llvm {
class Error;
class Module;
} // namespace llvm

namespace ops_mlir {

inline constexpr const char *kPersistentCudaStreamGetterName =
    "ops_mlir_get_persistent_cuda_stream";

llvm::Error useMonoCudaStream(llvm::Module *llvmModule);

} // namespace ops_mlir

#endif // OPS_MLIR_PASSES_USE_MONO_CU_STREAM_H
