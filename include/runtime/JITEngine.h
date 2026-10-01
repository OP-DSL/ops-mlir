#ifndef OPS_CAPTURE_H
#define OPS_CAPTURE_H

#include "IRBuilder.h"
#include "Core.h"
#include "runtime/FusionPlanner.h"
#include "runtime/KernelIRBuilder.h"
#include "runtime/KernelProfiler.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>


namespace ops_mlir {

enum class Backend { Sequential, OpenMP, CUDA };

std::optional<Backend> parseBackendName(const std::string &name);
constexpr Backend kDefaultBackend = Backend::CUDA;

static constexpr const char *kBackendFlagPrefix = "--backend=";
static constexpr const char *kBackendEnvVar = "OPS_BACKEND";

struct XdslResult {
  bool success = false;
  std::string ir;      // populated on success
  std::string error;   // populated on failure
};

class ModuleKey {
public:
  // `planDigest` (FusionPlan::digest) keeps modules compiled for different
  // groupings of the same loops apart.
  explicit ModuleKey(const std::vector<LoopDesc> &queue,
                     const std::string &planDigest = "") {
    digest_ += "plan:" + planDigest + "\n";
    for (const LoopDesc &loop : queue) {
      digest_ += loop.kernel_name;
      digest_ += '|';
      digest_ += std::to_string(loop.dims);
      digest_ += '|';
      for (int64_t r : loop.range) {
        digest_ += std::to_string(r);
        digest_ += ',';
      }
      for (const ArgDesc &arg : loop.args) {
        digest_ += '[';
        digest_ += std::to_string(arg.argtype);
        digest_ += ';';
        digest_ += std::to_string(arg.acc);
        digest_ += ';';
        digest_ += std::to_string(arg.dim);
        digest_ += ';';
        digest_ += std::to_string(arg.elem_size);
        digest_ += ';';
        digest_ += std::to_string(arg.opt);
        digest_ += ';';
        digest_ += std::to_string(arg.elem_kind);

        const DatDesc &dat = arg.dat;
        digest_ += ";dat:";
        digest_ += std::to_string(dat.dim);
        digest_ += ',';
        digest_ += std::to_string(dat.type_size);
        digest_ += ',';
        digest_ += std::to_string(dat.elem_size);
        digest_ += ',';
        digest_ += dat.type;
        for (int64_t v : dat.size) { digest_ += ','; digest_ += std::to_string(v); }
        for (int64_t v : dat.base) { digest_ += ','; digest_ += std::to_string(v); }
        for (int64_t v : dat.d_m) { digest_ += ','; digest_ += std::to_string(v); }
        for (int64_t v : dat.d_p) { digest_ += ','; digest_ += std::to_string(v); }
        for (int64_t v : dat.stride) { digest_ += ','; digest_ += std::to_string(v); }

        const StencilDesc &st = arg.stencil;
        digest_ += ";st:";
        digest_ += std::to_string(st.dims);
        digest_ += ',';
        digest_ += std::to_string(st.points);
        digest_ += ',';
        digest_ += std::to_string(st.type);
        if (st.stencil) {
          const int *offsets = reinterpret_cast<const int *>(st.stencil);
          for (int i = 0; i < st.dims * st.points; ++i) {
            digest_ += ',';
            digest_ += std::to_string(offsets[i]);
          }
        }
        digest_ += ']';
      }
      digest_ += '\n';
    }
  }

  [[nodiscard]] const std::string &digest() const { return digest_; }

  bool operator==(const ModuleKey &other) const {
    return digest_ == other.digest_;
  }

  [[nodiscard]] std::size_t hash() const {
    return std::hash<std::string>{}(digest_);
  }

private:
  std::string digest_;
};

} // namespace ops_mlir

namespace std {
template <> struct hash<ops_mlir::ModuleKey> {
  std::size_t operator()(const ops_mlir::ModuleKey &key) const {
    return key.hash();
  }
};
} // namespace std

namespace ops_mlir {

class BackendPipeline;

class JITEngine {
public:
  using FlushCallback = std::function<void(const std::vector<LoopDesc> &)>;

  static JITEngine &instance();

  void setFlushCallback(FlushCallback callback);

  void enqueueParLoop(std::uintptr_t kernelToken, const char *kernelName,
                      ops_block block, int dims, const int *range,
                      const ops_arg *args, std::size_t nargs,
                      std::function<void()> fallback = {},
                      const int *elemKinds = nullptr);

  void flush();

  void compile_and_execute();

  // Runs any queued loops. Cheap no-op when the queue is empty. Everything
  // that lets the host observe or modify OPS data goes through this first
  // (see the interception macros in ops/OPSWrapper.h).
  void flushPending();

  // flushPending() plus a device->host copy of `dat` when it lives on the GPU.
  void hostAccess(ops_dat dat);
  void hostAccessAll();

  // Counters for tests and profiling.
  struct Stats {
    std::size_t numCompiles = 0; // module cache misses
    std::size_t numFlushes = 0;  // non-empty queue flushes
    std::size_t numLoops = 0;    // par_loops executed
    std::size_t numLaunches = 0; // generated functions invoked (one per group)
    std::size_t numHostLoops = 0; // loops run through the stock OPS fallback
    // Wall-clock seconds spent in the runtime (for profiling).
    double compileSeconds = 0;   // planning is cheap; this is IR build + lowering + JIT
    double xdslSeconds = 0;      //   of which: the Python (xDSL) lowering
    double kernelIRSeconds = 0;  //   of which: translating kernel bodies
    double backendSeconds = 0;   //   of which: the MLIR backend pipeline
    double engineSeconds = 0;    //   of which: LLVM translation and code generation
    double executeSeconds = 0;   // launches, including the kernels themselves
    double kernelSeconds = 0;    // just invokePacked + device sync, summed over launches
    double haloSeconds = 0;      // ops_halo_transfer, excluding any flush it triggers
    double enqueueSeconds = 0;   // ops_par_loop: describing the loop (dats, stencils, globals)
    double planSeconds = 0;      // per flush: fusion plan + module-cache key + lookup
  };
  const Stats &stats() const { return stats_; }
  Stats &mutableStats() { return stats_; }
  void setQueueMax(std::size_t n) { queueMax_ = n; }

  // Loop fusion (see runtime/FusionPlanner.h). Defaults come from the
  // environment; tests override them.
  void setFusionOptions(const FusionOptions &o) { fusionOptions_ = o; }
  const FusionOptions &fusionOptions() const { return fusionOptions_; }
  // Grouping used by the most recent flush.
  const FusionPlan &lastPlan() const { return lastPlan_; }
  void resetStats() { stats_ = Stats(); }

  void setBackend(Backend backend) { backend_ = backend; }
  Backend backend() const { return backend_; }

  // Needed for GPU backend kernel translation (via KernelIRBuilder) to materialize real MLIR
  // May be called once per kernel header; a kernel is looked up in all of them.
  void setKernelSourceFile(std::string path) {
    for (const std::string &f : kernelSourceFiles_)
      if (f == path)
        return;
    kernelSourceFiles_.push_back(std::move(path));
  }

  // Source text parsed in front of every kernel file by the accessor translator.
  void setKernelPreamble(std::string text) { kernelPreamble_ = std::move(text); }

  // Register extern global kernel constants (e.g. pi, jmax) for translation into MLIR 
  void registerKernelConstant(const std::string &name, const void *ptr) {
    kernelConstants_[name] = ptr;
  }

  const std::map<std::string, const void *> &kernelConstants() const {
    return kernelConstants_;
  }

  // Note - resolveBackend is currently unused. This gives the option of a
  Backend resolveBackend(int argc, char **argv);

  const std::vector<LoopDesc> &queue() const { return queue_; }

  KernelProfiler &profiler() { return profiler_; }

private:
  JITEngine();
  ~JITEngine();

  mlir::MLIRContext ctx;
  mlir::ModuleOp module;
  mlir::OwningOpRef<mlir::ModuleOp> loweredModule_;

  IRBuilder builder{&ctx};

  LoopDesc buildLoopDesc(std::uintptr_t kernelToken, const char *kernelName,
                         ops_block block, int dims, const int *range,
                         const ops_arg *args, std::size_t nargs,
                         const int *elemKinds);

  ArgDesc buildArgDesc(const ops_arg &arg, int elemKind);

  // Appends the current value of every registered constant the loop's kernel reads
  // as a synthetic read-only global, so it reaches the compiled code as an argument.
  void attachConstantArgs(LoopDesc &loop);

  DatDesc describeDat(ops_dat dat);
  StencilDesc describeStencil(ops_stencil stencil);

  XdslResult runXdslLowering(const std::string &ir);

  void runBackendLowering(mlir::ModuleOp module, Backend backend);
  std::string detectNVGpuSm();

  void reportPlan(const ModuleKey &key, const FusionPlan &plan);
  void compile(const FusionPlan &plan);

  // Runs the queue's JIT-compilable stretch (see compile_and_execute).
  void compileAndExecuteSegment();
  // Loops with a stock-OPS fallback run on the host when the JIT can't compile
  // them (OPS_MLIR_HOST=all forces it for every such loop, =none never uses it).
  bool runsOnHost(const LoopDesc &loop);
  void runHostLoop(const LoopDesc &loop);
  // OPS_MLIR_VERIFY=1: run one JIT-compiled loop, then redo it with the stock
  // implementation from the same starting data and report any difference.
  void verifyLoop(LoopDesc &loop);
  bool verify_ = false;
  std::map<std::string, std::size_t> verifyBad_, verifyRuns_;
  bool kernelTranslatable(const LoopDesc &loop);
  // What each argument of an accessor-style kernel means (see KernelArgInfo).
  std::vector<KernelArgInfo> kernelArgInfos(const LoopDesc &loop);
  static std::string argInfoDigest(const std::vector<KernelArgInfo> &infos);
  void syncHostBufferPtr(std::uintptr_t hostPtr);
  enum class HostMode { Auto, All, None };
  HostMode hostMode_ = HostMode::Auto;
  struct Probe { bool ok; std::string signature; std::vector<KernelConstRef> constRefs; };
  std::map<std::string, Probe> translatable_; // by kernel name; first signature wins
  bool explain_ = false;                       // OPS_MLIR_EXPLAIN
  std::map<std::string, double> hostSeconds_; // fallback time per kernel
  void execute(mlir::ExecutionEngine &engine, const FusionPlan &plan);

  // Name of the generated function for group `gid`; must match
  // group_func_name in xdsl_impl/ops_to_stencil.py.
  std::string groupFunctionName(const FusedGroup &group, std::size_t gid) const;
  void registerCpuKernelSymbols(mlir::ExecutionEngine &engine);

  // Translates a kernel body from C++ source into MLIR via KernelIRBuilder,
  // splicing it in to replace the `func.func private @kernel(...)`
  // declaration xDSL lowering leaves behind -- required for CUDA (device
  // code can't call back into host object code) and used for the CPU
  // backends too so the kernel body can be inlined into the surrounding
  // loop rather than staying an opaque call through a symbol bound by
  // registerCpuKernelSymbols. Returns false (leaving the declaration in
  // place) if the kernel isn't already materialized and translation fails
  // or is unsupported, so callers can fall back to symbol binding.
  bool materializeKernelBody(const LoopDesc &loop);

  // Returns the device buffer mirroring the given host `ops_dat` buffer,
  // allocating it (via cuMemAlloc) on first use. Kernels compiled for the
  // CUDA backend operate on device memory, not the host pointers `dat`
  // args normally carry, so execute() copies host->device before each
  // launch and device->host after for any dat the kernel writes. A no-op
  // returning 0 when built without OPS_ENABLE_CUDA.
  //
  // Note: unconditionally declared (not #ifdef OPS_ENABLE_CUDA) even
  // though only meaningful for CUDA builds -- OPS_ENABLE_CUDA is only
  // defined PRIVATE for the OPSRuntime target, so consumer translation
  // units (e.g. apps linking against it) never see that macro; gating a
  // class member on it here would make JITEngine's layout depend on
  // which TU compiles it, an ODR violation.
  std::uintptr_t ensureDeviceBuffer(std::uintptr_t hostPtr, std::size_t bytes);

  void synchronizeBackend(Backend backend);

public:

  void invalidateDeviceBuffer(std::uintptr_t hostPtr);

  void syncHostBuffer(ops_dat dat);

  void syncAllHostBuffers();

  bool haloTransferDevice(ops_halo_group group);

  void shutdown();

  // Returns a persistent CUDA stream (CUstream) for kernels to launch on.
  std::uintptr_t ensurePersistentCudaStream();

private:
  Backend backend_ = kDefaultBackend;
  std::size_t queueMax_ = 512; // OPS_MLIR_QUEUE_MAX; auto-flush at this length
  Stats stats_;
  FusionOptions fusionOptions_ = FusionOptions::fromEnv();
  FusionPlan lastPlan_;
  std::set<std::string> reportedPlans_;
  std::mutex mutex_;
  std::vector<LoopDesc> queue_;
  FlushCallback flushCallback_;
  std::vector<std::string> kernelSourceFiles_;
  std::string kernelPreamble_;
  std::map<std::string, const void *> kernelConstants_;

  std::unordered_map<ModuleKey, std::unique_ptr<mlir::ExecutionEngine>>
      engineCache_;

  std::unique_ptr<BackendPipeline> currentPipeline_;

  KernelProfiler profiler_;

  // Host ops_dat pointer -> cached device buffer (stored as uintptr_t to
  // keep this header CUDA-toolkit-header-free; only populated/used when
  // built with OPS_ENABLE_CUDA).
  //  - `dirty`: the device copy is stale relative to the host buffer and
  //    must be re-copied (HtoD) before the device buffer is next used --
  //    set by invalidateDeviceBuffer() (host-side halo fallback).
  //  - `hostDirty`: the host copy is stale relative to the device buffer
  //    and must be re-copied (DtoH) before host code reads dat->data --
  //    set after a kernel writes the dat, or after a device-resident halo
  //    exchange writes it; cleared lazily by syncHostBuffer().
  struct DeviceBufferEntry {
    std::uintptr_t devPtr;
    std::size_t bytes = 0;
    bool dirty = false;
    bool hostDirty = false;
  };
  std::map<std::uintptr_t, DeviceBufferEntry> deviceBuffers_;
};

// Forwards to the real ops_halo_transfer (linked in from the OPS host
// library) -- unless the CUDA backend can service the whole group on-device
// (see JITEngine::haloTransferDevice), in which case the host copy is
// skipped entirely and dats keep flowing GPU-only. OPSWrapper.h #defines
// ops_halo_transfer to route call sites through this.
void haloTransferIntercepted(ops_halo_group group);

void exitIntercepted();

const char *accessToString(int access);
const char *argKindToString(ArgKind kind);

} // namespace ops_mlir

#endif // OPS_CAPTURE_H
