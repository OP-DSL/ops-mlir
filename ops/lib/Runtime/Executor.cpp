#include "ops-mlir/Runtime/Executor.h"

#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"
#include "ops-mlir/Pipelines/GroupModules.h"
#include "ops-mlir/Pipelines/Pipelines.h"
#include "ops-mlir/Runtime/CallDump.h"
#include "ops-mlir/Runtime/Claim.h"
#include "ops-mlir/Runtime/Lifter.h"

#include "opdsl/GPU/GpuShim.h"

#include "opdsl/Support/Hash.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include "ops_lazy_executor.h"

#include <cmath>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>

namespace ops_mlir::runtime {

using namespace mlir;
using namespace ops_mlir::ops;
using Clock = std::chrono::steady_clock;

namespace {

double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

struct ScopedTime {
  double &sink;
  Clock::time_point t0 = Clock::now();
  explicit ScopedTime(double &s) : sink(s) {}
  ~ScopedTime() { sink += since(t0); }
};

/// Collects the diagnostics emitted while alive.
struct CaptureDiagnostics {
  std::string text;
  ScopedDiagnosticHandler handler;
  explicit CaptureDiagnostics(MLIRContext *ctx)
      : handler(ctx, [this](Diagnostic &d) {
          llvm::raw_string_ostream os(text);
          os << d << "\n";
          return success();
        }) {}
};

std::string errorText(llvm::Error e) { return llvm::toString(std::move(e)); }

/// OPS_JIT_DUMP_DIR: writes `text` (or the module) to <dir>/<name>.
void dumpText(const std::string &name, llvm::StringRef text) {
  const std::string &dir = options().dumpDir;
  if (dir.empty()) return;
  llvm::sys::fs::create_directories(dir);
  std::error_code ec;
  llvm::raw_fd_ostream os(dir + "/" + name, ec);
  if (!ec) os << text;
}

void dumpModule(const std::string &name, ModuleOp module) {
  if (options().dumpDir.empty()) return;
  std::string text;
  llvm::raw_string_ostream os(text);
  module->print(os);
  dumpText(name, text);
}

/// The module of one group without what makes equal code look different: loop numbers, concrete ranges.
OwningOpRef<ModuleOp> canonicalGroup(ModuleOp m, bool keepRanges) {
  OwningOpRef<ModuleOp> copy(cast<ModuleOp>(m->clone()));
  Builder b(m.getContext());
  for (ChainOp chain : copy->getOps<ChainOp>()) {
    chain.setSymName("group");
    if (!keepRanges) chain->removeAttr("ops.range_values");
    chain->removeAttr("ops.const_values");
    chain->removeAttr("ops.const_roles");
    chain->removeAttr("ops.binding_consts");
    chain.walk([&](ParLoopOp loop) {
      loop->setAttr("loop_id", b.getI64IntegerAttr(-1));
      loop->setAttr("block", b.getI64IntegerAttr(0));
      loop->removeAttr("ops.fused_loops");
    });
  }
  return copy;
}

/// slot = slot (kind) value, with OPS's semantics of MIN/MAX (a < b ? a : b).
template <typename T> void combineInto(char *slot, T value, ReductionKind kind) {
  T cur;
  std::memcpy(&cur, slot, sizeof(T));
  T next;
  if (kind == ReductionKind::Add) next = cur + value;
  else if (kind == ReductionKind::Min) next = cur < value ? cur : value;
  else next = cur > value ? cur : value;
  std::memcpy(slot, &next, sizeof(T));
}


/// Everything OPS_JIT_VERIFY needs to know about one group before and after it ran.
struct Snapshot {
  std::vector<ops_dat> dats;               ///< dats the members write
  std::vector<std::vector<char>> before, jit;
  std::vector<std::pair<char *, size_t>> slots; ///< reduction handles the members update
  std::vector<std::vector<char>> slotBefore, slotJit;
};

std::vector<char> copyBytes(const char *p, size_t n) { return std::vector<char>(p, p + n); }

/// Largest |a-b| and largest |a|, |b| over `n` elements of type T.
template <typename T> void compareElements(const char *a, const char *b, size_t n, double &maxDiff, double &maxAbs, size_t &differing) {
  for (size_t i = 0; i < n; ++i) {
    T x, y;
    std::memcpy(&x, a + i * sizeof(T), sizeof(T));
    std::memcpy(&y, b + i * sizeof(T), sizeof(T));
    if (x == y || (x != x && y != y)) {
      double ax = std::fabs((double)x);
      if (ax == ax && ax > maxAbs) maxAbs = ax;
      continue;
    }
    ++differing;
    double d = std::fabs((double)x - (double)y);
    if (d != d) d = HUGE_VAL;
    if (d > maxDiff) maxDiff = d;
    maxAbs = std::max(maxAbs, std::max(std::fabs((double)x), std::fabs((double)y)));
  }
}

/// Relative difference (to the largest magnitude in the data) of two buffers of `type` elements, and the number of
/// elements that differ.
double relativeDifference(const char *type, const char *a, const char *b, size_t bytes, size_t &differing) {
  double maxDiff = 0, maxAbs = 0;
  differing = 0;
  std::string t = type ? type : "";
  if (t == "double") compareElements<double>(a, b, bytes / 8, maxDiff, maxAbs, differing);
  else if (t == "float") compareElements<float>(a, b, bytes / 4, maxDiff, maxAbs, differing);
  else compareElements<int32_t>(a, b, bytes / 4, maxDiff, maxAbs, differing);
  if (maxDiff == 0) return 0;
  return maxAbs > 0 ? maxDiff / maxAbs : maxDiff;
}

double toleranceFor(const char *type) {
  std::string t = type ? type : "";
  if (t == "double") return 1e-12;
  if (t == "float") return 1e-5;
  return 0;
}


/// The whole allocation of a dat as bytes, from the host or the device copy.
std::vector<char> readDat(ops_dat d, bool device) {
  std::vector<char> bytes(d->mem);
  if (device) mgpuMemcpy(bytes.data(), d->data_d, d->mem, nullptr);
  else std::memcpy(bytes.data(), d->data, d->mem);
  return bytes;
}

void writeDat(ops_dat d, const std::vector<char> &bytes, bool device) {
  if (device) mgpuMemcpy(d->data_d, const_cast<char *>(bytes.data()), bytes.size(), nullptr);
  else std::memcpy(d->data, bytes.data(), bytes.size());
}

/// The thread block for a block of `rank` dimensions: the option (x,y,z), with the dimensions the block does not have folded into x.
std::array<int64_t, 3> gpuBlockFor(const std::string &option, int rank) {
  std::array<int64_t, 3> b{32, 4, 1};
  std::sscanf(option.c_str(), "%ld,%ld,%ld", &b[0], &b[1], &b[2]);
  for (int d = rank; d < 3; ++d) {
    b[0] *= b[d];
    b[d] = 1;
  }
  return b;
}

/// Number of elements of a reduction buffer: one per warp (or per thread when the block is not whole warps) of the grid that the
/// lowering launches for the ranges `lo/hi` (see convert-ops-to-scf, target gpu).
int64_t gpuReductionUnits(const std::array<int64_t, 3> &block, const int64_t *extent, int rank) {
  int64_t blocks = 1;
  for (int d = 0; d < rank; ++d) blocks *= std::max<int64_t>(0, (std::max<int64_t>(extent[d], 0) + block[d] - 1) / block[d]);
  const int64_t threads = block[0] * block[1] * block[2];
  const int64_t perBlock = threads % 32 == 0 ? threads / 32 : threads;
  return std::max<int64_t>(1, blocks * perBlock);
}

/// The PTX version the toolkit's ptxas accepts, as an NVVM target feature (+ptx83 for CUDA 12.3): newer PTX allows larger kernel
/// parameter blocks (32 KB from PTX 8.1) which the fused groups of big applications need.
std::string ptxFeature() {
  static std::string feature = [] {
    std::string cmd = "ptxas --version 2>/dev/null";
    const char *root = std::getenv("CUDA_ROOT");
    if (!root) root = std::getenv("CUDA_HOME");
    if (!root) root = std::getenv("CUDA_PATH");
    if (root) cmd = std::string("'") + root + "/bin/ptxas' --version 2>/dev/null";
    std::string out;
    if (FILE *p = popen(cmd.c_str(), "r")) {
      char buf[256];
      while (std::fgets(buf, sizeof buf, p)) out += buf;
      pclose(p);
    }
    size_t at = out.find("release ");
    int major = 0, minor = 0;
    if (at != std::string::npos && std::sscanf(out.c_str() + at, "release %d.%d", &major, &minor) == 2 && major == 12)
      return "+ptx" + std::to_string(80 + std::min(minor, 7));
    return std::string("+ptx80");
  }();
  return feature;
}

} // namespace

Executor::Executor() {
  registerOPSPipelines(registry_);
  registerJITDialects(dialects_);
  ctx_ = std::make_unique<MLIRContext>(dialects_, MLIRContext::Threading::DISABLED);
  ctx_->loadAllAvailableDialects();
  artifacts_ = std::make_unique<ArtifactCache>(*ctx_);
  CompilerOptions co;
  co.useCache = options().cache;
  co.cacheDir = options().cacheDir;
  co.dumpDir = options().dumpDir;
  co.jit.fpContract = options().fma;
  compiler_ = std::make_unique<GroupCompiler>(*ctx_, registry_, co);
}

Executor &executor() {
  static Executor *instance = new Executor(); // never destroyed: JIT'd code and OPS may outlive static destructors
  return *instance;
}

void Executor::run(OPS_instance *instance, ops_kernel_descriptor **descs, int n, void *ctx) {
  (void)ctx;
  executor().execute(instance, descs, n);
}

bool Executor::deviceReady() {
  if (deviceState_ < 0) {
    deviceState_ = opdsl::gpu::cudaAvailable() ? 1 : 0;
    if (deviceState_) {
      // The NVVM target finds ptxas and libdevice through the toolkit path: the usual variables, else the CUDA used to build OPS.
      if (!std::getenv("CUDA_ROOT") && !std::getenv("CUDA_HOME") && !std::getenv("CUDA_PATH"))
        if (const char *root = std::getenv("CUDA_INSTALL_PATH")) ::setenv("CUDA_ROOT", root, 0);
    }
  }
  return deviceState_ == 1;
}

void *Executor::deviceScratch(size_t bytes) {
  if (bytes > scratchBytes_) {
    if (scratch_) mgpuMemFree(scratch_, nullptr);
    scratchBytes_ = std::max<size_t>(bytes, 256);
    scratch_ = mgpuMemAlloc(scratchBytes_, nullptr, false);
  }
  return scratch_;
}

void Executor::execute(OPS_instance *instance, ops_kernel_descriptor **descs, int n) {
  const Options &opts = options();
  stats_.flushes++;
  std::vector<const KernelArtifact *> artifacts(n, nullptr);
  std::vector<char> claimed(n, 0);
  {
    ScopedTime timer(stats_.tClaim);
    static std::set<std::pair<std::string, std::string>> explained;
    for (int i = 0; i < n; ++i) {
      stats_.loopsSeen++;
      std::string reason;
      if (opts.claim == "none") reason = "OPS_JIT_CLAIM=none";
      else if (descs[i]->startup_func) reason = "the loop has a startup function";
      else {
        artifacts[i] = artifacts_->get(descs[i]->jit_artifact, reason);
        if (artifacts[i]) reason = claimReason(descs[i], *artifacts[i], descs[i]->isdevice ? deviceReady() : false);
      }
      if (reason.empty()) {
        claimed[i] = 1;
        stats_.loopsClaimed++;
      } else {
        stats_.loopsDeclined++;
        stats_.declineReasons[reason]++;
        if (opts.explain && explained.insert({descs[i]->name, reason}).second)
          std::fprintf(stderr, "ops_jit: %s runs with OPS: %s\n", descs[i]->name, reason.c_str());
      }
    }
  }

  int i = 0;
  while (i < n) {
    if (!claimed[i]) {
      int j = i;
      while (j < n && !claimed[j]) ++j;
      ops_lazy_execute_default(instance, descs + i, j - i);
      i = j;
      continue;
    }
    int j = i;
    while (j < n && claimed[j] && j - i < opts.runMax && descs[j]->isdevice == descs[i]->isdevice) ++j; // host and device loops do not share a plan
    std::vector<const KernelArtifact *> runArtifacts(artifacts.begin() + i, artifacts.begin() + j);
    if (!runJit(instance, descs + i, j - i, runArtifacts)) ops_lazy_execute_default(instance, descs + i, j - i);
    i = j;
  }
}

std::string Executor::structuralKey(ops_kernel_descriptor **run, int n, const std::vector<std::vector<int>> &ranges,
                                    const std::vector<const KernelArtifact *> &artifacts, bool soa) {
  opdsl::Hasher h;
  h.add(options().codegenDigest()).addBool(soa).addInt(kCodeFormat);
  std::map<int, int> slotOf;
  std::set<std::string> boundDone;
  for (int k = 0; k < n; ++k) {
    ops_kernel_descriptor *d = run[k];
    h.add(artifacts[k]->manifest.digest).addInt(d->block->index).addInt(d->dim).addInt(d->isdevice);
    for (int a = 0; a < d->nargs; ++a) {
      const ops_arg &arg = d->args[a];
      h.addInt(arg.argtype).addInt(arg.acc);
      if (arg.argtype == OPS_ARG_DAT) {
        int slot = slotOf.emplace(arg.dat->index, (int)slotOf.size()).first->second;
        h.addInt(slot).add(arg.dat->type ? arg.dat->type : "").addInt(arg.dat->dim).addInt(arg.dat->type_size).addInt(arg.dat->e_dat);
        h.addInt(arg.stencil->points).addInt(arg.stencil->type);
        for (int p = 0; p < arg.stencil->points * arg.stencil->dims; ++p) h.addInt(arg.stencil->stencil[p]);
        for (int dd = 0; dd < arg.stencil->dims; ++dd) h.addInt(arg.stencil->stride[dd]);
      } else if (arg.argtype == OPS_ARG_GBL) {
        h.addInt(arg.dim).addInt(arg.elem_size);
      }
    }
    for (int v : ranges[k]) h.addInt(v);
    for (const ConstInfo &c : artifacts[k]->manifest.consts) {
      if (!c.bound || !boundDone.insert(c.name).second) continue;
      const ops_const_entry *e = ops_find_const(d->block->instance, c.name.c_str());
      if (e) h.add(llvm::StringRef(e->snapshot, static_cast<size_t>(e->dim) * e->size));
    }
  }
  return h.hex();
}

bool Executor::runJit(OPS_instance *instance, ops_kernel_descriptor **descs, int n, const std::vector<const KernelArtifact *> &artifactsIn) {
  const Options &opts = options();

  // Local ranges. A loop with nothing to do on this process is skipped altogether, as OPS does.
  std::vector<ops_kernel_descriptor *> run;
  std::vector<std::vector<int>> ranges;
  std::vector<const KernelArtifact *> artifacts;
  for (int k = 0; k < n; ++k) {
    int local[2 * OPS_MAX_DIM] = {0};
    if (!ops_lazy_compute_local_range(descs[k], local)) continue;
    run.push_back(descs[k]);
    ranges.emplace_back(local, local + 2 * descs[k]->block->dims);
    artifacts.push_back(artifactsIn[k]);
  }
  const int m = static_cast<int>(run.size());
  if (m == 0) return true;

  const bool soa = ops_lazy_soa(instance) != 0;
  std::string key = structuralKey(run.data(), m, ranges, artifacts, soa);
  std::shared_ptr<Plan> plan;
  auto it = plans_.find(key);
  if (it != plans_.end()) {
    plan = it->second;
    stats_.planHits++;
  } else {
    ScopedTime timer(stats_.tPlan);
    plan = buildPlan(run.data(), m, ranges, artifacts, key);
    plans_[key] = plan;
  }
  if (!plan->declined.empty()) return false;

  std::vector<ops_dat> slots = datSlots(run.data(), m);
  for (Plan::Group &g : plan->groups) {
    if (!g.code) {
      // Not compiled: the loops of the group run with OPS's code, in order.
      for (int k : g.recipe.members) ops_lazy_execute_default(instance, &run[k], 1);
      continue;
    }
    Snapshot snap;
    if (opts.verify) {
      std::set<ops_dat> seen;
      for (int k : g.recipe.members)
        for (int a = 0; a < run[k]->nargs; ++a) {
          const ops_arg &arg = run[k]->args[a];
          if (arg.argtype == OPS_ARG_DAT && arg.acc != OPS_READ && seen.insert(arg.dat).second) snap.dats.push_back(arg.dat);
          if (arg.argtype == OPS_ARG_GBL && arg.acc != OPS_READ) {
            char *slot = ops_lazy_reduction_slot(run[k], a);
            snap.slots.push_back({slot, (size_t)arg.dim * elemBytes(elemName(((ops_reduction)arg.data)->type))});
          }
        }
      for (int k : g.recipe.members) ops_lazy_prepare(run[k]);
      for (ops_dat d : snap.dats) snap.before.push_back(readDat(d, g.device));
      for (auto &s : snap.slots) snap.slotBefore.push_back(copyBytes(s.first, s.second));
    } else {
      for (int k : g.recipe.members) ops_lazy_prepare(run[k]);
    }

    // Device groups return their reductions through device buffers (8 bytes each), initialised with the neutral element.
    std::vector<void *> reductionBuffers;
    std::vector<char> staging;
    int64_t units = 0;
    if (g.device) {
      const size_t count = g.recipe.reductionOf.size();
      int64_t extent[3] = {1, 1, 1};
      for (int d = 0; d < g.recipe.rank; ++d) {
        const ChainArg &lo = g.recipe.range[2 * d], &hi = g.recipe.range[2 * d + 1];
        extent[d] = ranges[hi.loop][2 * hi.dim + 1] - ranges[lo.loop][2 * lo.dim];
      }
      units = gpuReductionUnits(gpuBlockFor(opts.gpuBlock, g.recipe.rank), extent, g.recipe.rank);
      staging.assign(std::max<size_t>(count, 1) * units * 8, 0);
      char *base = static_cast<char *>(deviceScratch(staging.size()));
      for (size_t r = 0; r < count; ++r) reductionBuffers.push_back(base + r * units * 8);
    }

    Invocation inv;
    std::string error;
    auto tb = Clock::now();
    if (!inv.bind(g.recipe, run.data(), ranges, slots, soa, error, g.device ? &reductionBuffers : nullptr, units)) {
      std::fprintf(stderr, "ops_jit: binding failed: %s\n", error.c_str());
      std::abort(); // loops of the run may already have executed: no way back
    }
    stats_.tBind += since(tb);
    if (opts.trace) {
      std::fprintf(stderr, "ops_jit: running %s:", g.code->entry.c_str());
      for (int k : g.recipe.members) std::fprintf(stderr, " %s", run[k]->name);
      std::fprintf(stderr, "\n");
    }
    std::unique_ptr<CallRecorder> recorder;
    if (opts.dumpData && !opts.dumpDir.empty() && dumpedGroups_.insert(g.code->entry).second) {
      std::vector<std::string> elems;
      for (int slot : g.recipe.datSlots) elems.push_back(elemName(slots[slot]->type ? slots[slot]->type : ""));
      recorder = std::make_unique<CallRecorder>(opts.dumpDir, g.code->entry, g.code->pipeline, g.code->pipelineOptions, inv.args(), elems);
    }
    auto t0 = Clock::now();
    inv.call(g.code->fn);
    if (g.device && !reductionBuffers.empty()) {
      // The copy waits for the kernel (default stream) and for the host to have the values.
      mgpuMemcpy(staging.data(), reductionBuffers.front(), staging.size(), nullptr);
    }
    double seconds = since(t0);
    if (opts.trace) std::fprintf(stderr, "ops_jit:   %.6f s\n", seconds);
    stats_.tExecute += seconds;
    stats_.groupsExecuted++;
    if (recorder) {
      std::string rerror;
      if (!recorder->finish(rerror)) std::fprintf(stderr, "ops_jit: cannot record %s: %s\n", g.code->entry.c_str(), rerror.c_str());
    }

    for (size_t r = 0; r < g.recipe.reductionOf.size(); ++r) {
      int index = g.recipe.reductionOf[r];
      if (index < 0) continue;
      const ReductionRef &ref = plan->reductions[index];
      char *slot = ops_lazy_reduction_slot(run[ref.loop], ref.arg);
      if (g.device) {
        // The value is the combination of the partial values of the warps in the staging copy of the device buffer.
        for (int64_t k = 0; k < units; ++k) {
          const char *p = &staging[(r * units + k) * 8];
          if (ref.elem == "f64") {
            double v;
            std::memcpy(&v, p, 8);
            combineInto<double>(slot, v, ref.kind);
          } else if (ref.elem == "f32") {
            float v;
            std::memcpy(&v, p, 4);
            combineInto<float>(slot, v, ref.kind);
          } else {
            int32_t v;
            std::memcpy(&v, p, 4);
            combineInto<int32_t>(slot, v, ref.kind);
          }
        }
      } else if (ref.elem == "f64") combineInto<double>(slot, inv.resultAsDouble(r), ref.kind);
      else if (ref.elem == "f32") combineInto<float>(slot, static_cast<float>(inv.resultAsDouble(r)), ref.kind);
      else combineInto<int32_t>(slot, static_cast<int32_t>(inv.resultAsInt(r)), ref.kind);
    }

    const double share = seconds / std::max<size_t>(1, g.recipe.members.size());
    for (int k : g.recipe.members) {
      ops_lazy_finish(run[k]);
      if (instance->OPS_diags > 1) {
        std::vector<int> local = ranges[k];
        local.resize(2 * OPS_MAX_DIM, 0);
        ops_lazy_account(run[k], share, ops_lazy_transfer_bytes(run[k], local.data()));
      }
    }

    if (opts.verify) {
      for (ops_dat d : snap.dats) snap.jit.push_back(readDat(d, g.device));
      for (auto &s : snap.slots) snap.slotJit.push_back(copyBytes(s.first, s.second));
      // Back to the state before the group, then the stock code for each member in order.
      for (size_t i = 0; i < snap.dats.size(); ++i) writeDat(snap.dats[i], snap.before[i], g.device);
      for (size_t i = 0; i < snap.slots.size(); ++i) std::memcpy(snap.slots[i].first, snap.slotBefore[i].data(), snap.slotBefore[i].size());
      for (int k : g.recipe.members) ops_lazy_execute_default(instance, &run[k], 1);
      stats_.verifyGroups++;
      std::string names;
      for (int k : g.recipe.members) names += std::string(names.empty() ? "" : ", ") + run[k]->name;
      bool bad = false;
      for (size_t i = 0; i < snap.dats.size(); ++i) {
        size_t differing = 0;
        std::vector<char> aot = readDat(snap.dats[i], g.device);
        double rel = relativeDifference(snap.dats[i]->type, snap.jit[i].data(), aot.data(), snap.jit[i].size(), differing);
        if (rel > toleranceFor(snap.dats[i]->type)) {
          bad = true;
          std::fprintf(stderr, "ops_jit verify: [%s] dat %s: %zu elements differ, relative difference %.3g (tolerance %.1g)\n", names.c_str(),
                       snap.dats[i]->name ? snap.dats[i]->name : "?", differing, rel, toleranceFor(snap.dats[i]->type));
        }
      }
      for (size_t i = 0; i < snap.slots.size(); ++i) {
        // Reduction values: compared in relative terms; the handles are doubles, floats or ints.
        size_t bytes = snap.slotJit[i].size();
        size_t differing = 0;
        const char *type = bytes == 8 ? "double" : bytes == 4 ? "float" : "int";
        double rel = relativeDifference(type, snap.slotJit[i].data(), snap.slots[i].first, bytes, differing);
        if (rel > std::max(toleranceFor(type), 1e-9)) {
          bad = true;
          auto asDouble = [&](const char *p) {
            if (bytes == 8) { double v; std::memcpy(&v, p, 8); return v; }
            if (bytes == 4 && type[0] == 'f') { float v; std::memcpy(&v, p, 4); return double(v); }
            int v; std::memcpy(&v, p, 4); return double(v);
          };
          std::fprintf(stderr, "ops_jit verify: [%s] reduction %zu differs, relative difference %.3g (before %.17g, JIT %.17g, OPS %.17g)\n",
                       names.c_str(), i, rel, asDouble(snap.slotBefore[i].data()), asDouble(snap.slotJit[i].data()), asDouble(snap.slots[i].first));
        }
      }
      if (bad) stats_.verifyMismatches++;
    }
  }
  return true;
}

std::shared_ptr<Plan> Executor::buildPlan(ops_kernel_descriptor **run, int n, const std::vector<std::vector<int>> &ranges,
                                          const std::vector<const KernelArtifact *> &artifacts, const std::string &planKey) {
  const Options &opts = options();
  stats_.planBuilds++;
  auto plan = std::make_shared<Plan>();
  auto fail = [&](const std::string &why) {
    plan->declined = why;
    stats_.planFailures++;
    stats_.declineReasons["plan: " + why]++;
    if (opts.explain) std::fprintf(stderr, "ops_jit: a run of %d loops starting with %s runs with OPS: %s\n", n, run[0]->name, why.c_str());
    return plan;
  };

  CaptureDiagnostics diagnostics(ctx_.get());
  Lifted lifted;
  std::string error;
  {
    ScopedTime timer(stats_.tLift);
    if (!liftRun(*ctx_, run, n, ranges, artifacts, opts.printfPolicy == "keep", lifted, error)) return fail("lifting: " + error);
  }
  {
    ScopedTime timer(stats_.tFrontend);
    PassManager pm(ctx_.get());
    if (llvm::Error e = registry_.build("ops-jit-frontend", "", pm)) return fail(errorText(std::move(e)));
    if (failed(pm.run(*lifted.module))) return fail("ops-jit-frontend failed: " + diagnostics.text);
    if (!recordBindingConsts(lifted, error)) return fail(error);
  }
  {
    ScopedTime timer(stats_.tPlanPasses);
    PassManager pm(ctx_.get());
    std::string planOptions = opts.fusion ? "max-loops=" + std::to_string(opts.fusionMax) +
                                                " reorder=" + (opts.fusionReorder ? "true" : "false") +
                                                " placement=" + opts.fusionPlacement + " guarded=" + (opts.fusionGuarded ? "true" : "false")
                                          : "max-loops=1";
    if (llvm::Error e = registry_.build("ops-jit-plan", planOptions, pm)) return fail(errorText(std::move(e)));
    if (failed(pm.run(*lifted.module))) return fail("ops-jit-plan failed: " + diagnostics.text);
  }

  dumpModule("plan_" + planKey.substr(0, 16) + ".mlir", *lifted.module);
  std::vector<GroupModule> groups = extractGroupModules(*lifted.module);
  std::vector<GroupRecipe> recipes;
  if (!buildRecipes(lifted, *lifted.module, recipes, error)) return fail(error);
  if (groups.size() != recipes.size()) return fail("internal: groups and recipes disagree");
  const bool soa = ops_lazy_soa(run[0]->block->instance) != 0;
  for (size_t g = 0; g < groups.size(); ++g) {
    OwningOpRef<ModuleOp> canonical = canonicalGroup(groups[g].module.get(), opts.specialize != "none");
    std::string cerror;
    const bool device = run[0]->isdevice != 0;
    std::string pipeline, pipelineOptions;
    if (device) {
      // block sizes with `x` between them, ptxas options with `:` between them (the option syntax uses blanks and commas)
      std::array<int64_t, 3> bs = gpuBlockFor(opts.gpuBlock, run[0]->block->dims);
      std::string block = std::to_string(bs[0]) + "x" + std::to_string(bs[1]) + "x" + std::to_string(bs[2]), ptxas = opts.ptxasOpts;
      std::replace(ptxas.begin(), ptxas.end(), ' ', ':');
      std::string arch = opts.gpuArch.empty() ? opdsl::gpu::cudaDeviceArch() : opts.gpuArch;
      pipeline = "ops-jit-cuda";
      pipelineOptions = "chip=" + arch + " features=" + ptxFeature() + " block=" + block + " specialize=" + opts.specialize + " soa=" + (soa ? "true" : "false");
      if (!ptxas.empty()) pipelineOptions += " ptxas=" + ptxas;
    } else {
      pipeline = "ops-jit-cpu";
      pipelineOptions = std::string("omp=") + (opts.omp ? "true" : "false") + " specialize=" + opts.specialize + " soa=" + (soa ? "true" : "false");
    }
    std::shared_ptr<GroupCode> code = compiler_->compile(canonical.get(), pipeline, pipelineOptions, cerror);
    Plan::Group pg;
    pg.recipe = std::move(recipes[g]);
    pg.device = device;
    if (!code) {
      // Only this group runs with OPS (e.g. a GPU kernel with more parameters than the driver allows).
      stats_.groupsDeclined++;
      std::string firstLine = cerror.substr(0, cerror.find('\n'));
      stats_.declineReasons["group: " + firstLine]++;
      if (opts.explain) {
        std::fprintf(stderr, "ops_jit: a group of %zu loops starting with %s runs with OPS: compiling failed: %s\n", pg.recipe.members.size(),
                     run[pg.recipe.members.front()]->name, cerror.c_str());
      }
    }
    pg.code = std::move(code);
    plan->groups.push_back(std::move(pg));
  }
  plan->reductions = lifted.reductions;

  if (opts.plan) {
    std::fprintf(stderr, "ops_jit: plan for %d loops: %zu groups\n", n, plan->groups.size());
    for (size_t g = 0; g < plan->groups.size(); ++g) {
      std::fprintf(stderr, "  group %zu [%s]:", g, plan->groups[g].code ? plan->groups[g].code->entry.c_str() : "OPS");
      for (int k : plan->groups[g].recipe.members) std::fprintf(stderr, " %s", run[k]->name);
      std::fprintf(stderr, "\n");
    }
  }
  return plan;
}

void Executor::printStats() const {
  const Stats &s = stats_;
  const CompileStats &c = compiler_->stats();
  std::fprintf(stderr, "ops_jit counters: flushes=%llu loops=%llu claimed=%llu declined=%llu groups=%llu plans_built=%llu plans_reused=%llu plans_failed=%llu groups_declined=%llu compiled=%llu memory_hits=%llu disk_hits=%llu verify_groups=%llu verify_mismatches=%llu\n",
               (unsigned long long)s.flushes, (unsigned long long)s.loopsSeen, (unsigned long long)s.loopsClaimed,
               (unsigned long long)s.loopsDeclined, (unsigned long long)s.groupsExecuted, (unsigned long long)s.planBuilds,
               (unsigned long long)s.planHits, (unsigned long long)s.planFailures, (unsigned long long)s.groupsDeclined, (unsigned long long)c.compiles,
               (unsigned long long)c.memoryHits, (unsigned long long)c.diskHits, (unsigned long long)s.verifyGroups,
               (unsigned long long)s.verifyMismatches);
  std::fprintf(stderr, "ops_jit statistics:\n");
  std::fprintf(stderr, "  flushes %llu, loops %llu: claimed %llu, declined %llu; groups executed %llu\n", (unsigned long long)s.flushes,
               (unsigned long long)s.loopsSeen, (unsigned long long)s.loopsClaimed, (unsigned long long)s.loopsDeclined,
               (unsigned long long)s.groupsExecuted);
  std::fprintf(stderr, "  plans: %llu built, %llu reused, %llu failed; code: %llu compiled, %llu from memory, %llu from disk\n",
               (unsigned long long)s.planBuilds, (unsigned long long)s.planHits, (unsigned long long)s.planFailures,
               (unsigned long long)c.compiles, (unsigned long long)c.memoryHits, (unsigned long long)c.diskHits);
  std::fprintf(stderr, "  time [s]: claim %.3f, lift %.3f, frontend %.3f, plan passes %.3f, lowering %.3f, llvm %.3f, cache io %.3f, bind %.3f, execute %.3f\n",
               s.tClaim, s.tLift, s.tFrontend, s.tPlanPasses, c.tLowering, c.tLlvm, c.tCacheIO, s.tBind, s.tExecute);
  if (options().verify)
    std::fprintf(stderr, "  verify: %llu groups checked against OPS, %llu with differences\n", (unsigned long long)s.verifyGroups,
                 (unsigned long long)s.verifyMismatches);
  for (const auto &entry : s.declineReasons)
    std::fprintf(stderr, "  declined %llux: %s\n", (unsigned long long)entry.second, entry.first.c_str());
}

} // namespace ops_mlir::runtime
