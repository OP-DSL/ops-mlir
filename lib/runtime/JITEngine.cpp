//===- JITEngine.cpp - JIT compilation/execution engine for OPS loops ---===//
//
// Part of OPS-MLIR Project
// Author: Prakanth Thilakaraj
// Date: June 2026
//
// This file is distributed under the MIT License.
// See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

#include "runtime/JITEngine.h"
#include "runtime/BackendPipeline.h"
#include "runtime/KernelIRBuilder.h"
#include "runtime/KernelProfiler.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Python.h"

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Host.h"

#include <algorithm>
#include <chrono>
#include <array>
#include <cstring>
#include <memory>
#include <set>

#ifdef OPS_ENABLE_CUDA
#include <cuda.h>
#endif

namespace {
// Adds the wall-clock time of its scope to a counter.
struct ScopedSeconds {
  explicit ScopedSeconds(double &sink)
      : sink_(sink), start_(std::chrono::steady_clock::now()) {}
  ~ScopedSeconds() {
    sink_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  }
  double &sink_;
  std::chrono::steady_clock::time_point start_;
};
} // namespace

namespace ops_mlir {

JITEngine &JITEngine::instance() {
  static JITEngine rt;
  return rt;
}

JITEngine::JITEngine() {
  if (!Py_IsInitialized()) {
    Py_Initialize();
  }

  // Make xdsl_impl/ops_to_xdsl.py importable. OPS_XDSL_DIR is injected by
  // lib/runtime/CMakeLists.txt as the absolute path to xdsl_impl/.
  std::string setup = "import sys\nsys.path.insert(0, '" OPS_XDSL_DIR "')\n";
  PyRun_SimpleString(setup.c_str());

  mlir::registerAllPasses();
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  ctx.getOrLoadDialect<mlir::omp::OpenMPDialect>();
  ctx.getOrLoadDialect<mlir::gpu::GPUDialect>();
  ctx.getOrLoadDialect<mlir::NVVM::NVVMDialect>();
  ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  ctx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  ctx.getOrLoadDialect<mlir::math::MathDialect>();

  // Register Interfaces
  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::registerAllExtensions(registry);
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  ctx.appendDialectRegistry(registry);

  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeNativeTargetAsmPrinter();

  // Resolve backend (without working CLI flags for now)
  backend_ = resolveBackend(0, nullptr);

  if (const char *cap = std::getenv("OPS_MLIR_QUEUE_MAX"))
    queueMax_ = static_cast<std::size_t>(std::strtoull(cap, nullptr, 10));
}

JITEngine::~JITEngine() {
  // Safe here specifically because profiler_ is a member subobject, not a
  // separate singleton -- see its declaration's comment (JITEngine.h).
  profiler_.report();
  if (std::getenv("OPS_MLIR_STATS"))
    llvm::errs() << "ops-mlir stats: " << stats_.numLoops << " loops, "
                 << stats_.numLaunches << " kernel launches, "
                 << stats_.numFlushes << " flushes, " << stats_.numCompiles
                 << " module compiles; seconds: compile "
                 << llvm::format("%.4f", stats_.compileSeconds) << " execute "
                 << llvm::format("%.4f", stats_.executeSeconds) << " halo "
                 << llvm::format("%.4f", stats_.haloSeconds) << " enqueue "
                 << llvm::format("%.4f", stats_.enqueueSeconds) << " plan "
                 << llvm::format("%.4f", stats_.planSeconds) << "\n";

  if (Py_IsInitialized()) {
    Py_FinalizeEx();
  }
}

void JITEngine::setFlushCallback(FlushCallback callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  flushCallback_ = std::move(callback);
}

void JITEngine::enqueueParLoop(std::uintptr_t kernelToken,
                                    const char *kernelName, ops_block block,
                                    int dims, const int *range,
                                    const ops_arg *args, std::size_t nargs) {
  bool full;
  {
    ScopedSeconds timer(stats_.enqueueSeconds);
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(
        buildLoopDesc(kernelToken, kernelName, block, dims, range, args, nargs));
    full = queueMax_ && queue_.size() >= queueMax_;
  }
  // Bound the size of a compiled module (and the memory held by the queue).
  if (full)
    flushPending();
}

void JITEngine::flushPending() {
  if (!queue_.empty())
    compile_and_execute();
}

void JITEngine::hostAccess(ops_dat dat) {
  flushPending();
  syncHostBuffer(dat);
}

void JITEngine::hostAccessAll() {
  flushPending();
  syncAllHostBuffers();
}

void JITEngine::flush() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (flushCallback_ && !queue_.empty()) {
    flushCallback_(queue_);
  }
  queue_.clear();
}

std::string JITEngine::detectNVGpuSm() {
  if (const char *env = std::getenv("OPS_GPU_SM"))
    return env;

#ifdef OPS_ENABLE_CUDA
  if (cuInit(0) != CUDA_SUCCESS) {
    throw std::runtime_error(
        "cuInit failed -- no NVIDIA driver found. Set OPS_GPU_SM manually (e.g. \"sm_86\").");
  }
  CUdevice device;
  cuDeviceGet(&device, 0);
  int major = 0, minor = 0;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  return "sm_" + std::to_string(major) + std::to_string(minor);
#else
  throw std::runtime_error(
      "This build was compiled without CUDA support (OPS_ENABLE_CUDA=OFF). "
      "Reconfigure with -DOPS_ENABLE_CUDA=ON to use the CUDA backend, or "
      "set OPS_GPU_SM manually (e.g. \"sm_86\") if targeting a remote/"
      "precompiled binary.");
#endif
}

LoopDesc JITEngine::buildLoopDesc(std::uintptr_t kernelToken,
                                       const char *kernelName, ops_block block,
                                       int dims, const int *range,
                                       const ops_arg *args, std::size_t nargs) {
  LoopDesc loop;
  loop.kernel_name = kernelName;
  loop.kernel_token = kernelToken;
  loop.block = reinterpret_cast<std::uintptr_t>(block);
  loop.dims = dims;

  loop.range.assign(range, range + 2 * dims);

  loop.args.reserve(nargs);
  for (std::size_t i = 0; i < nargs; ++i) {
    loop.args.push_back(buildArgDesc(args[i]));
  }

  return loop;
}

ArgDesc JITEngine::buildArgDesc(const ops_arg &arg) {
  ArgDesc desc;
  desc.dim = arg.dim;
  desc.elem_size = arg.elem_size;
  desc.data = reinterpret_cast<std::uintptr_t>(arg.data);
  desc.data_d = reinterpret_cast<std::uintptr_t>(arg.data_d);
  desc.acc = arg.acc;
  desc.argtype = arg.argtype;
  desc.opt = arg.opt;

  if (arg.argtype == OPS_ARG_GBL && arg.acc == OPS_READ && arg.data) {
    std::size_t nbytes = static_cast<std::size_t>(arg.elem_size) *
                         static_cast<std::size_t>(std::max(arg.dim, 1));
    desc.gbl_value.assign(arg.data, arg.data + nbytes);
  }

  if (arg.argtype == OPS_ARG_DAT || arg.argtype == OPS_ARG_GBL) {
    if (arg.dat)
      desc.dat = describeDat(arg.dat);
    if (arg.stencil)
      desc.stencil = describeStencil(arg.stencil);
  }

  return desc;
}

DatDesc JITEngine::describeDat(ops_dat dat) {
  DatDesc d;
  d.handle = reinterpret_cast<std::uintptr_t>(dat); // TODO: Casting pointers to uintptr_t is not portable. We should use a better way to represent pointers in MLIR attributes.
  d.index = dat->index;
  d.block = reinterpret_cast<std::uintptr_t>(dat->block); // TODO: Casting pointers to uintptr_t is not portable. We should use a better way to represent pointers in MLIR attributes.
  
  d.dim = dat->dim;
  d.type_size = dat->type_size;
  d.elem_size = dat->elem_size;

  int ndim = dat->block ? dat->block->dims : 1;
  for (int i = 0; i < ndim; ++i) {
    d.size.push_back(dat->size[i]);
    d.base.push_back(dat->base[i]);
    d.d_m.push_back(dat->d_m[i]);
    d.d_p.push_back(dat->d_p[i]);
    d.stride.push_back(dat->stride[i]);
  }

  d.name = dat->name ? dat->name : "";
  d.type = dat->type ? dat->type : "";

  d.data = reinterpret_cast<std::uintptr_t>(dat->data);
  d.data_d = reinterpret_cast<std::uintptr_t>(dat->data_d);

  return d;
}

StencilDesc JITEngine::describeStencil(ops_stencil stencil) {
  StencilDesc s;
  s.index = stencil->index;
  s.dims = stencil->dims;
  s.points = stencil->points;
  s.name = stencil->name ? stencil->name : "";

  s.stencil = reinterpret_cast<std::uintptr_t>(stencil->stencil);
  s.stride = reinterpret_cast<std::uintptr_t>(stencil->stride);
  s.mgrid_stride = reinterpret_cast<std::uintptr_t>(stencil->mgrid_stride);
  s.type = stencil->type;

  return s;
}

static std::string fetchPyError() {
  if (!PyErr_Occurred())
    return "unknown Python error";
  PyObject *type, *value, *tb;
  PyErr_Fetch(&type, &value, &tb);
  PyErr_NormalizeException(&type, &value, &tb);
  std::string msg = "unknown Python error";
  if (value) {
    PyObject *str = PyObject_Str(value);
    if (str) {
      if (const char *s = PyUnicode_AsUTF8(str)) msg = s;
      Py_DECREF(str);
    }
  }
  Py_XDECREF(type); Py_XDECREF(value); Py_XDECREF(tb);
  return msg;
}

XdslResult JITEngine::runXdslLowering(const std::string &ir) {
  PyGILState_STATE gstate = PyGILState_Ensure();
  XdslResult result;

  PyObject *mod = PyImport_ImportModule("ops_to_xdsl");
  if (!mod) {
    result.error = fetchPyError();
    PyGILState_Release(gstate);
    return result;
  }

  PyObject *func = PyObject_GetAttrString(mod, "convert_ir_text");
  Py_DECREF(mod);
  if (!func || !PyCallable_Check(func)) {
    result.error = fetchPyError();
    Py_XDECREF(func);
    PyGILState_Release(gstate);
    return result;
  }

  PyObject *args = Py_BuildValue("(s)", ir.c_str());
  if (!args) {
    result.error = fetchPyError();
    Py_DECREF(func);
    PyGILState_Release(gstate);
    return result;
  }
  PyObject *pyResult = PyObject_CallObject(func, args);
  Py_DECREF(args);
  Py_DECREF(func);

  if (!pyResult) {
    result.error = fetchPyError();
    PyGILState_Release(gstate);
    return result;
  }

  if (const char *text = PyUnicode_AsUTF8(pyResult)) {
    result.ir = text;
    result.success = true;
  } else {
    result.error = fetchPyError();
  }
  Py_DECREF(pyResult);

  PyGILState_Release(gstate);
  return result;
}

void JITEngine::runBackendLowering(mlir::ModuleOp module, Backend backend) {
  std::unique_ptr<BackendPipeline> pipeline;

  switch (backend) {
  case Backend::Sequential:
    pipeline = std::make_unique<CPUSequentialPipeline>();
    break;
  case Backend::OpenMP:
    pipeline = std::make_unique<OpenMPPipeline>();
    break;
  case Backend::CUDA:
#ifdef OPS_ENABLE_CUDA
    pipeline = std::make_unique<CudaPipeline>(detectNVGpuSm());
#else
    throw std::runtime_error(
        "CUDA backend requested but this build was compiled without "
        "CUDA support (OPS_ENABLE_CUDA=OFF).");
#endif
    break;
  }

  if (!pipeline) {
    llvm::errs() << "no lowering pipeline for requested backend\n";
    return;
  }

  if (mlir::failed(pipeline->run(module, ctx))) {
    llvm::errs() << "backend lowering failed for module\n";
    return;
  }

  // Store the pipeline for later use.
  currentPipeline_ = std::move(pipeline);

#ifdef OPS_ENABLE_DEBUG
  llvm::outs() << "=== BACKEND-LOWERED MLIR IR ===\n\n";
  module.print(llvm::outs());
  llvm::outs() << "\n";
  llvm::outs().flush();
#endif
}

void JITEngine::compile(const FusionPlan &plan) {
  std::vector<int64_t> groupIds(queue_.size(), 0);
  for (std::size_t g = 0; g < plan.groups.size(); ++g)
    for (std::size_t l : plan.groups[g].loops)
      groupIds[l] = static_cast<int64_t>(g);

  module = builder.buildModule(queue_, &groupIds);

  std::string ir = builder.moduleToString(module);

#ifdef OPS_ENABLE_DEBUG
  llvm::outs() << "=== OPS.PAR_LOOP MLIR IR ===\n\n" << ir << "\n";
  llvm::outs().flush();
#endif

  if (std::getenv("OPS_MLIR_DUMP_LOWERED"))
    llvm::errs() << "=== OPS.PAR_LOOP IR ===\n" << ir << "\n";

  XdslResult lowered = runXdslLowering(ir);
  if (!lowered.success) {
    llvm::errs() << "xDSL lowering failed: " <<  lowered.error << "\n";
    return;
  }
  
#ifdef OPS_ENABLE_DEBUG
  llvm::outs() << "=== LOWERED STENCIL IR (xDSL, in-process) ===\n\n"
            << lowered.ir << "\n";
  llvm::outs().flush();
#endif

  // OPS_MLIR_DUMP_LOWERED=1: print the per-group IR coming out of xDSL.
  if (std::getenv("OPS_MLIR_DUMP_LOWERED")) {
    llvm::errs() << "=== LOWERED IR (fusion plan " << plan.digest()
                 << ") ===\n" << lowered.ir << "\n";
  }

  loweredModule_ =
    mlir::parseSourceString<mlir::ModuleOp>(lowered.ir, &ctx);

  if (!loweredModule_) {
    llvm::errs() << "Failed to parse xDSL output as MLIR\n";
    return;
  }

  std::vector<std::pair<std::string, int>> kernels;
  for (const LoopDesc &loop : queue_) {
    bool seen = std::any_of(
        kernels.begin(), kernels.end(),
        [&](const auto &k) { return k.first == loop.kernel_name; });
    if (!seen)
      kernels.emplace_back(loop.kernel_name, loop.dims);
  }
  for (const auto &[name, dims] : kernels) {
    materializeKernelBody(name, dims);
  }

  runBackendLowering(*loweredModule_, backend_);
  module = *loweredModule_;
}

bool JITEngine::materializeKernelBody(const std::string &kernelName,
                                      int indexRank) {
  if (kernelSourceFile_.empty()) {
    llvm::errs() << "materializeKernelBody: no kernel source file set "
                    "(call setKernelSourceFile), cannot translate '"
                 << kernelName << "'\n";
    return false;
  }

  mlir::func::FuncOp declOp;
  loweredModule_->walk([&](mlir::func::FuncOp fn) {
    if (fn.getSymName() == kernelName && fn.isDeclaration())
      declOp = fn;
  });
  if (!declOp) {
    // Already materialized (e.g. a prior loop with the same kernel name),
    // or not present in this module.
    return true;
  }

  KernelIRBuilder kernelBuilder(ctx);
  mlir::func::FuncOp translatedFn = kernelBuilder.generate(
      kernelSourceFile_, kernelName, indexRank, kernelConstants_, llvm::errs());
  if (!translatedFn) {
    llvm::errs() << "materializeKernelBody: could not translate '"
                 << kernelName << "' from " << kernelSourceFile_ << "\n";
    return false;
  }

  declOp.erase();
  loweredModule_->push_back(translatedFn);
  return true;
}


static std::size_t datByteSize(const DatDesc &dat) {
  std::size_t total = static_cast<std::size_t>(dat.elem_size);
  for (int64_t dim : dat.size)
    total *= static_cast<std::size_t>(dim);
  return total;
}

static double computeDataTransfer(const LoopDesc &loop, const ArgDesc &arg) {
  const int *stencilStride =
      reinterpret_cast<const int *>(arg.stencil.stride);

  double size = 1.0;
  for (int i = 0; i < loop.dims; ++i) {
    int64_t extent = loop.range[2 * i + 1] - loop.range[2 * i];
    bool moves = !stencilStride || stencilStride[i] != 0;
    if (moves && extent > 0)
      size *= static_cast<double>(extent);
  }
  size *= arg.dat.elem_size *
          ((arg.acc == OPS_READ || arg.acc == OPS_WRITE) ? 1.0 : 2.0);
  return size;
}

static double computeDataTransferPerLoop(const LoopDesc &loop) {
  double bytes = 0.0;
  for (const ArgDesc &arg : loop.args) {
    if (arg.argtype == OPS_ARG_DAT)
      bytes += computeDataTransfer(loop, arg);
  }
  return bytes;
}

std::uintptr_t JITEngine::ensureDeviceBuffer(std::uintptr_t hostPtr,
                                             std::size_t bytes) {
#ifdef OPS_ENABLE_CUDA
  auto it = deviceBuffers_.find(hostPtr);
  if (it != deviceBuffers_.end()) {
    if (it->second.dirty) {
      cuMemcpyHtoD(static_cast<CUdeviceptr>(it->second.devPtr),
                  reinterpret_cast<const void *>(hostPtr), bytes);
      it->second.dirty = false;
    }
    return it->second.devPtr;
  }

  // Use the same (primary) CUDA context mlir_cuda_runtime's wrappers use
  // (see CudaRuntimeWrappers.cpp's ScopedContext), so buffers allocated
  // here are valid in the kernels that runtime launches.
  static bool contextReady = [] {
    cuInit(0);
    CUdevice device;
    cuDeviceGet(&device, 0);
    CUcontext ctx;
    cuDevicePrimaryCtxRetain(&ctx, device);
    cuCtxSetCurrent(ctx);
    return true;
  }();
  (void)contextReady;

  CUdeviceptr devPtr = 0;
  if (CUresult rc = cuMemAlloc(&devPtr, bytes); rc != CUDA_SUCCESS) {
    llvm::errs() << "ensureDeviceBuffer: cuMemAlloc failed (CUresult " << rc
                 << ")\n";
    return 0;
  }
  deviceBuffers_[hostPtr] = {static_cast<std::uintptr_t>(devPtr), bytes, false, false};
  cuMemcpyHtoD(devPtr, reinterpret_cast<const void *>(hostPtr), bytes);
  return devPtr;
#else
  (void)hostPtr;
  (void)bytes;
  return 0;
#endif
}

std::uintptr_t JITEngine::ensurePersistentCudaStream() {
#ifdef OPS_ENABLE_CUDA
  static CUstream stream = [] {
    CUstream s = nullptr;
    if (CUresult rc = cuStreamCreate(&s, CU_STREAM_DEFAULT); rc != CUDA_SUCCESS) {
      llvm::errs() << "ensurePersistentCudaStream: cuStreamCreate failed "
                      "(CUresult "
                   << rc << ")\n";
      return static_cast<CUstream>(nullptr);
    }
    return s;
  }();
  return reinterpret_cast<std::uintptr_t>(stream);
#else
  return 0;
#endif
}


extern "C" void *ops_mlir_get_persistent_cuda_stream() {
  return reinterpret_cast<void *>(JITEngine::instance().ensurePersistentCudaStream());
}

// TODO: Use dat.data_d instead of deviceBuffers_.
void JITEngine::invalidateDeviceBuffer(std::uintptr_t hostPtr) {
  auto it = deviceBuffers_.find(hostPtr);
  if (it != deviceBuffers_.end())
    it->second.dirty = true;
}

void JITEngine::syncHostBuffer(ops_dat dat) {
#ifdef OPS_ENABLE_CUDA
  auto it = deviceBuffers_.find(reinterpret_cast<std::uintptr_t>(dat->data));
  if (it == deviceBuffers_.end() || !it->second.hostDirty)
    return;
  cuMemcpyDtoH(reinterpret_cast<void *>(dat->data), it->second.devPtr,
              it->second.bytes);
  it->second.hostDirty = false;
#else
  (void)dat;
#endif
}

void JITEngine::syncAllHostBuffers() {
#ifdef OPS_ENABLE_CUDA
  for (auto &[hostPtr, entry] : deviceBuffers_) {
    if (!entry.hostDirty)
      continue;
    cuMemcpyDtoH(reinterpret_cast<void *>(hostPtr), entry.devPtr, entry.bytes);
    entry.hostDirty = false;
  }
#endif
}

#ifdef OPS_ENABLE_CUDA
static std::size_t opsDatByteSize(ops_dat dat) {
  std::size_t total = static_cast<std::size_t>(dat->elem_size);
  for (int i = 0; i < dat->block->dims; ++i)
    total *= static_cast<std::size_t>(dat->size[i]);
  return total;
}
#endif

// TODO: [Remove] Replace with OPS implementation of device halo transfer
bool JITEngine::haloTransferDevice(ops_halo_group group) {
#ifdef OPS_ENABLE_CUDA
  if (backend_ != Backend::CUDA)
    return false;

  for (int h = 0; h < group->nhalos; ++h) {
    ops_halo halo = group->halos[h];
    int dims = halo->from->block->dims;
    if (dims != 3)
      return false;
    for (int i = 0; i < dims; ++i)
      if (halo->from_dir[i] != i + 1 || halo->to_dir[i] != i + 1)
        return false;
    if (halo->from->type_size != halo->to->type_size ||
        halo->from->elem_size != halo->to->elem_size)
      return false;
  }

  for (int h = 0; h < group->nhalos; ++h) {
    ops_halo halo = group->halos[h];
    ops_dat from = halo->from;
    ops_dat to = halo->to;
    int dims = from->block->dims;
    int elemSize = from->elem_size;

    std::uintptr_t fromDev = ensureDeviceBuffer(
        reinterpret_cast<std::uintptr_t>(from->data), opsDatByteSize(from));
    std::uintptr_t toDev = ensureDeviceBuffer(
        reinterpret_cast<std::uintptr_t>(to->data), opsDatByteSize(to));
    if (!fromDev || !toDev)
      return false;

    long long fromStart[3] = {0, 0, 0}, toStart[3] = {0, 0, 0};
    long long extent[3] = {1, 1, 1};
    for (int i = 0; i < dims; ++i) {
      fromStart[i] = halo->from_base[i] - from->d_m[i] - from->base[i];
      toStart[i] = halo->to_base[i] - to->d_m[i] - to->base[i];
      extent[i] = halo->iter_size[i];
    }

    CUDA_MEMCPY3D cpy;
    memset(&cpy, 0, sizeof(cpy));
    cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cpy.srcDevice = static_cast<CUdeviceptr>(fromDev);
    cpy.srcXInBytes = static_cast<std::size_t>(fromStart[0]) * elemSize;
    cpy.srcY = static_cast<std::size_t>(fromStart[1]);
    cpy.srcZ = static_cast<std::size_t>(fromStart[2]);
    cpy.srcPitch = static_cast<std::size_t>(from->size[0]) * elemSize;
    cpy.srcHeight = static_cast<std::size_t>(from->size[1]);

    cpy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cpy.dstDevice = static_cast<CUdeviceptr>(toDev);
    cpy.dstXInBytes = static_cast<std::size_t>(toStart[0]) * elemSize;
    cpy.dstY = static_cast<std::size_t>(toStart[1]);
    cpy.dstZ = static_cast<std::size_t>(toStart[2]);
    cpy.dstPitch = static_cast<std::size_t>(to->size[0]) * elemSize;
    cpy.dstHeight = static_cast<std::size_t>(to->size[1]);

    cpy.WidthInBytes = static_cast<std::size_t>(extent[0]) * elemSize;
    cpy.Height = static_cast<std::size_t>(extent[1]);
    cpy.Depth = static_cast<std::size_t>(extent[2]);

    if (CUresult rc = cuMemcpy3D(&cpy); rc != CUDA_SUCCESS) {
      llvm::errs() << "haloTransferDevice: cuMemcpy3D failed (CUresult "
                   << rc << ")\n";
      return false;
    }

    auto it = deviceBuffers_.find(reinterpret_cast<std::uintptr_t>(to->data));
    if (it != deviceBuffers_.end()) {
      it->second.dirty = false;
      it->second.hostDirty = true;
    }
  }
  return true;
#else
  (void)group;
  return false;
#endif
}

void haloTransferIntercepted(ops_halo_group group) {
  // Halo exchange reads dats that queued loops may still have to produce.
  JITEngine::instance().flushPending();
  ScopedSeconds timer(JITEngine::instance().mutableStats().haloSeconds);
  if (JITEngine::instance().haloTransferDevice(group))
    return;

  for (int i = 0; i < group->nhalos; ++i)
    JITEngine::instance().syncHostBuffer(group->halos[i]->from);

  ::ops_halo_transfer(group);

  for (int i = 0; i < group->nhalos; ++i) {
    ops_dat to = group->halos[i]->to;
    JITEngine::instance().invalidateDeviceBuffer(
        reinterpret_cast<std::uintptr_t>(to->data));
  }
}

void JITEngine::shutdown() {
  engineCache_.clear();
}

void exitIntercepted() {
  JITEngine::instance().hostAccessAll();
  JITEngine::instance().shutdown();
  ::ops_exit();
}

void JITEngine::synchronizeBackend(Backend backend) {
  switch (backend) {
  case Backend::Sequential:
  case Backend::OpenMP:
    return;
  case Backend::CUDA:
#ifdef OPS_ENABLE_CUDA
    cuCtxSynchronize();
#endif
    return;
  }
}



// OPS_MLIR_PLAN=1: describe the kernels chosen for each distinct queue (once).
void JITEngine::reportPlan(const ModuleKey &key, const FusionPlan &plan) {
  if (!reportedPlans_.insert(key.digest()).second)
    return;
  TrafficEstimate traffic = estimateTraffic(queue_, plan);
  llvm::errs() << "[plan] " << queue_.size() << " loops -> "
               << plan.groups.size() << " kernels, " << plan.numReordered()
               << " loops moved, est. traffic "
               << llvm::format("%.3g", traffic.unfusedBytes) << " -> "
               << llvm::format("%.3g", traffic.fusedBytes) << " bytes ("
               << llvm::format("%.2f", traffic.fusedBytes
                                           ? traffic.unfusedBytes / traffic.fusedBytes
                                           : 1.0)
               << "x)\n";
  if (!plan.blockedBy.empty()) {
    llvm::errs() << "[plan]   new kernel started because:";
    for (const auto &[why, count] : plan.blockedBy)
      llvm::errs() << " " << why << "=" << count;
    llvm::errs() << "\n";
  }
  for (std::size_t g = 0; g < plan.groups.size(); ++g) {
    const FusedGroup &grp = plan.groups[g];
    llvm::errs() << "[plan]   K" << g << (grp.guarded ? " guarded" : "") << ":";
    for (std::size_t l : grp.loops)
      llvm::errs() << " " << queue_[l].kernel_name << "#" << l;
    llvm::errs() << "\n";
  }
}

std::string JITEngine::groupFunctionName(const FusedGroup &group,
                                         std::size_t gid) const {
  if (group.loops.size() == 1) {
    std::size_t loopIndex = group.loops.front();
    return "ops_par_loop_" + queue_[loopIndex].kernel_name + "_" +
           std::to_string(loopIndex);
  }
  return "ops_par_loop_group_" + std::to_string(gid);
}

/**
 * Execute the queued loops, one generated function per fusion group, using
 * the provided execution engine. Cache execution engines for previously
 * compiled modules to avoid recompilation.
 *
 * A group's function takes one buffer per distinct dat (by ops_dat index, in
 * first-appearance order over its member loops), then the read-only scalar
 * globals of every member loop in order -- the layout convert_group in
 * xdsl_impl/ops_to_stencil.py emits.
*/
void JITEngine::execute(mlir::ExecutionEngine &engine, const FusionPlan &plan) {
  lastPlan_ = plan;
  for (std::size_t gid = 0; gid < plan.groups.size(); ++gid) {
    const FusedGroup &group = plan.groups[gid];
    std::string funcName = groupFunctionName(group, gid);

    // Distinct dats, and which of them any member writes.
    std::vector<const ArgDesc *> datSlots;
    std::set<int> seen, written;
    for (std::size_t l : group.loops) {
      for (const ArgDesc &arg : queue_[l].args) {
        if (arg.argtype != OPS_ARG_DAT)
          continue;
        if (seen.insert(arg.dat.index).second)
          datSlots.push_back(&arg);
        if (arg.acc == OPS_WRITE || arg.acc == OPS_RW || arg.acc == OPS_INC)
          written.insert(arg.dat.index);
      }
    }

    std::vector<void *> datPtrs;
    std::vector<std::pair<std::uintptr_t, std::size_t>> writebacks;
    for (const ArgDesc *slot : datSlots) {
      const ArgDesc &arg = *slot;
      if (backend_ == Backend::CUDA) {
        std::size_t bytes = datByteSize(arg.dat);
        std::uintptr_t devPtr = ensureDeviceBuffer(arg.data, bytes);
        if (!devPtr) {
          llvm::errs() << "Failed to allocate device buffer for '"
                       << funcName << "'\n";
          this->flush();
          return;
        }
        datPtrs.push_back(reinterpret_cast<void *>(devPtr));
        if (written.count(arg.dat.index))
          writebacks.emplace_back(arg.data, bytes);
      } else {
        datPtrs.push_back(reinterpret_cast<void *>(arg.data));
      }
    }

    // Read-only scalar globals are passed by value as f32 or f64, chosen by
    // sizeof(T) (OPS records only the size), from the value captured when the
    // loop was enqueued. Reduction globals are ignored.
    std::vector<std::array<char, 8>> gblScratch;
    for (std::size_t l : group.loops) {
      for (const ArgDesc &arg : queue_[l].args) {
        if (arg.argtype != OPS_ARG_GBL || arg.acc != OPS_READ)
          continue;
        std::array<char, 8> value{};
        std::memcpy(value.data(), arg.gbl_value.data(),
                    std::min<std::size_t>(arg.gbl_value.size(), value.size()));
        gblScratch.push_back(value);
      }
    }

    llvm::SmallVector<void *> packedArgs;
    packedArgs.reserve(datPtrs.size() + gblScratch.size());

    for (void *&ptr : datPtrs) {
      packedArgs.push_back(&ptr);
    }
    for (std::array<char, 8> &v : gblScratch) {
      packedArgs.push_back(v.data());
    }

    std::string profileName = queue_[group.loops.front()].kernel_name;
    double bytesMoved = 0.0;
    for (std::size_t l : group.loops)
      bytesMoved += computeDataTransferPerLoop(queue_[l]);
    if (group.loops.size() > 1) {
      profileName = "fused(";
      for (std::size_t k = 0; k < group.loops.size(); ++k)
        profileName += (k ? "+" : "") + queue_[group.loops[k]].kernel_name;
      profileName += ")";
    }

#ifdef OPS_ENABLE_DEBUG
    llvm::outs() << "About to invoke '" << funcName << "'\n";
    llvm::outs().flush();
#endif
    auto kernelStart = profiler_.start();
    auto launchStart = std::chrono::steady_clock::now();
    ++stats_.numLaunches;
    stats_.numLoops += group.loops.size();
    if (auto err = engine.invokePacked(funcName, packedArgs)) {
      llvm::errs() << "Failed to invoke '" << funcName
                  << "': " << llvm::toString(std::move(err)) << "\n";
      this->flush();
      return;
    }
    synchronizeBackend(backend_);
    stats_.kernelSeconds += std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - launchStart).count();
    profiler_.end(profileName, kernelStart, bytesMoved);
#ifdef OPS_ENABLE_CUDA
    for (const auto &[hostPtr, bytes] : writebacks) {
      auto it = deviceBuffers_.find(hostPtr);
      if (it != deviceBuffers_.end()) {
        it->second.hostDirty = true;
        it->second.bytes = bytes;
      }
    }
#endif
  }
  this->flush();
}

void JITEngine::compile_and_execute() {
  if (queue_.empty())
    return;

  auto planStart = std::chrono::steady_clock::now();
  FusionPlan plan = planFusion(queue_, fusionOptions_);
  ModuleKey key(queue_, plan.digest());
  ++stats_.numFlushes;
  if (std::getenv("OPS_MLIR_PLAN"))
    reportPlan(key, plan);

  auto cached = engineCache_.find(key);
  stats_.planSeconds += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - planStart).count();
  if (cached != engineCache_.end()) {
    ScopedSeconds timer(stats_.executeSeconds);
    execute(*cached->second, plan);
    return;
  }

  ++stats_.numCompiles;
  auto compileStart = std::chrono::steady_clock::now();
  compile(plan);

  llvm::Triple targetTriple(llvm::sys::getDefaultTargetTriple());
  std::string targetLookupError;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, targetLookupError);
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  if (!target) {
    llvm::errs() << "Warning: could not look up target for '"
                 << targetTriple.str() << "': " << targetLookupError
                 << "; falling back to no target machine (no "
                    "auto-vectorization)\n";
  } else {
    llvm::SubtargetFeatures features;
    for (auto &[feature, enabled] : llvm::sys::getHostCPUFeatures())
      features.AddFeature(feature, enabled);

    llvm::TargetOptions targetOptions;
    targetMachine.reset(target->createTargetMachine(
        targetTriple, llvm::sys::getHostCPUName(), features.getString(),
        targetOptions, /*RM=*/std::nullopt, /*CM=*/std::nullopt,
        llvm::CodeGenOptLevel::Aggressive));
  }

  mlir::ExecutionEngineOptions engineOptions;
  auto optTransformer = mlir::makeOptimizingTransformer(
    /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/targetMachine.get());

  std::function<llvm::Error(llvm::Module *)> transformer =
      [this, optTransformer](llvm::Module *m) -> llvm::Error {
    if (currentPipeline_)
      if (auto err = currentPipeline_->transformLLVMModule(m))
        return err;
    return optTransformer(m);
  };
  engineOptions.transformer = transformer;

  std::string cudaRuntimePath;
  // ExecutionEngineOptions::sharedLibPaths is an ArrayRef, so the backing
  // storage must outlive ExecutionEngine::create() below.
  llvm::StringRef sharedLibs[1];
  if (backend_ == Backend::CUDA) {
    if (const char *path = std::getenv("OPS_MLIR_CUDA_RUNTIME")) {
      cudaRuntimePath = path;
      sharedLibs[0] = cudaRuntimePath;
      engineOptions.sharedLibPaths = sharedLibs;
    } else {
      llvm::errs() << "Warning: OPS_MLIR_CUDA_RUNTIME not set; GPU kernel "
                     "launches will fail to resolve mgpu* symbols. Set it "
                     "to the path of libmlir_cuda_runtime.so.\n";
    }
  }

  auto engineOrErr = mlir::ExecutionEngine::create(module, engineOptions);
  if (!engineOrErr) {
    llvm::errs() << "Failed to create ExecutionEngine: "
                  << llvm::toString(engineOrErr.takeError()) << "\n";
    return;
  }

  auto engine = std::move(*engineOrErr);

  // TODO: [For inlining] Kernel are generate from KernelIRBuilder for all target.
  // switch (backend_) {
  // case Backend::Sequential:
  // case Backend::OpenMP:
  //   registerCpuKernelSymbols(*engine);
  //   break;
  // case Backend::CUDA:
  //   // Kernels should be added to the Host side. CPU side symbols are not valid.
  //   break;
  // }

  if (backend_ == Backend::CUDA) {
    engine->registerSymbols([](llvm::orc::MangleAndInterner interner) {
      llvm::orc::SymbolMap symbolMap;
      symbolMap[interner("ops_mlir_get_persistent_cuda_stream")] = {
          llvm::orc::ExecutorAddr::fromPtr(&ops_mlir_get_persistent_cuda_stream),
          llvm::JITSymbolFlags::Exported};
      return symbolMap;
    });
  }

  mlir::ExecutionEngine &engineRef = *engine;
  engineCache_.emplace(std::move(key), std::move(engine));
  stats_.compileSeconds += std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - compileStart)
                               .count();
  ScopedSeconds timer(stats_.executeSeconds);
  execute(engineRef, plan);
}

void JITEngine::registerCpuKernelSymbols(mlir::ExecutionEngine &engine) {
  engine.registerSymbols([this](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbolMap;
    for (const LoopDesc &loop : queue_) {
      void *kernelPtr = reinterpret_cast<void *>(loop.kernel_token);
      symbolMap[interner(loop.kernel_name)] = {
          llvm::orc::ExecutorAddr::fromPtr(kernelPtr),
          llvm::JITSymbolFlags::Exported};
    }
    return symbolMap;
  });
}

const char *accessToString(int access) {
  switch (access) {
  case OPS_READ:
    return "READ";
  case OPS_WRITE:
    return "WRITE";
  case OPS_RW:
    return "RW";
  case OPS_INC:
    return "INC";
  case OPS_MIN:
    return "MIN";
  case OPS_MAX:
    return "MAX";
  default:
    return "UNKNOWN";
  }
}

const char *argKindToString(ArgKind kind) {
  switch (kind) {
  case ArgKind::Dat:
    return "Dat";
  case ArgKind::Gbl:
    return "Gbl";
  case ArgKind::Idx:
    return "Idx";
  case ArgKind::Reduce:
    return "Reduce";
  default:
    return "Unknown";
  }
}

std::optional<Backend> parseBackendName(const std::string &name) {
  if (name == "seq" || name == "sequential") return Backend::Sequential;
  if (name == "openmp" || name == "omp") return Backend::OpenMP;
  if (name == "cuda" || name == "nvgpu") return Backend::CUDA; 
  return std::nullopt;
}


Backend JITEngine::resolveBackend(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind(kBackendFlagPrefix, 0) == 0) {
      std::string value = arg.substr(std::string(kBackendFlagPrefix).size());
      if (auto b = parseBackendName(value)) return *b;
      throw std::runtime_error("Unknown --backend value: '" + value + "' (expected seq|openmp|cuda)");
    }
  }

  // Fall back to env variable
  if (const char *env = std::getenv(kBackendEnvVar)) {
    if (auto b = parseBackendName(env)) return *b;
    throw std::runtime_error("Unknown " + std::string(kBackendEnvVar) + " value: '" + env + "' (expected seq|openmp|cuda)");
  }

  return kDefaultBackend;
}

} // namespace ops_mlir
