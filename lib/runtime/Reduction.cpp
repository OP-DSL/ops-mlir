//===- Reduction.cpp - Folding per-point contributions to one value ------===//
//
// See include/runtime/Reduction.h.
//===----------------------------------------------------------------------===//

#include "runtime/Reduction.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace ops_mlir {

namespace {

constexpr int kInc = 3, kMin = 4, kMax = 5; // OPS_INC, OPS_MIN, OPS_MAX

template <typename T> T identityOf(int access) {
  if (access == kInc)
    return T(0);
  constexpr bool inf = std::numeric_limits<T>::has_infinity;
  if (access == kMin)
    return inf ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();
  return inf ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest();
}

template <typename T> inline T combine(int access, T a, T v) {
  if (access == kInc)
    return static_cast<T>(a + v);
  if (access == kMin)
    return v < a ? v : a;
  return v > a ? v : a;
}

// Chunks accumulate in order (four independent lanes inside a chunk for speed, fixed lane
// assignment), and the chunk results are combined in order: the answer does not depend on how many
// threads ran.
template <typename T> T foldHost(const T *p, std::size_t n, int access, bool parallel) {
  constexpr std::size_t kChunks = 256;
  const std::size_t chunk = (n + kChunks - 1) / kChunks;
  std::vector<T> part(kChunks, identityOf<T>(access));
#pragma omp parallel for schedule(static) if (parallel)
  for (long c = 0; c < static_cast<long>(kChunks); ++c) {
    const std::size_t lo = std::min<std::size_t>(c * chunk, n), hi = std::min<std::size_t>(lo + chunk, n);
    T a0 = identityOf<T>(access), a1 = a0, a2 = a0, a3 = a0;
    std::size_t i = lo;
    for (; i + 4 <= hi; i += 4) {
      a0 = combine(access, a0, p[i]);
      a1 = combine(access, a1, p[i + 1]);
      a2 = combine(access, a2, p[i + 2]);
      a3 = combine(access, a3, p[i + 3]);
    }
    for (; i < hi; ++i)
      a0 = combine(access, a0, p[i]);
    part[c] = combine(access, combine(access, a0, a1), combine(access, a2, a3));
  }
  T r = identityOf<T>(access);
  for (T v : part)
    r = combine(access, r, v);
  return r;
}

template <typename T> void fillHost(T *p, std::size_t n, T v, bool parallel) {
#pragma omp parallel for schedule(static) if (parallel)
  for (long i = 0; i < static_cast<long>(n); ++i)
    p[i] = v;
}

bool validAccess(int access) { return access >= kInc && access <= kMax; }

} // namespace

bool reductionIdentity(int kind, int access, void *out8) {
  if (!validAccess(access))
    return false;
  std::memset(out8, 0, 8);
  switch (kind) {
  case EK_F32: { float v = identityOf<float>(access); std::memcpy(out8, &v, 4); return true; }
  case EK_F64: { double v = identityOf<double>(access); std::memcpy(out8, &v, 8); return true; }
  case EK_I32: { std::int32_t v = identityOf<std::int32_t>(access); std::memcpy(out8, &v, 4); return true; }
  case EK_I64: { std::int64_t v = identityOf<std::int64_t>(access); std::memcpy(out8, &v, 8); return true; }
  default: return false;
  }
}

bool reductionFillHost(void *buffer, std::size_t n, int kind, int access, bool parallel) {
  if (!validAccess(access))
    return false;
  switch (kind) {
  case EK_F32: fillHost(static_cast<float *>(buffer), n, identityOf<float>(access), parallel); return true;
  case EK_F64: fillHost(static_cast<double *>(buffer), n, identityOf<double>(access), parallel); return true;
  case EK_I32: fillHost(static_cast<std::int32_t *>(buffer), n, identityOf<std::int32_t>(access), parallel); return true;
  case EK_I64: fillHost(static_cast<std::int64_t *>(buffer), n, identityOf<std::int64_t>(access), parallel); return true;
  default: return false;
  }
}

bool reductionFoldHost(const void *buffer, std::size_t n, int kind, int access, bool parallel,
                       void *out8) {
  if (!validAccess(access))
    return false;
  std::memset(out8, 0, 8);
  switch (kind) {
  case EK_F32: { float v = foldHost(static_cast<const float *>(buffer), n, access, parallel); std::memcpy(out8, &v, 4); return true; }
  case EK_F64: { double v = foldHost(static_cast<const double *>(buffer), n, access, parallel); std::memcpy(out8, &v, 8); return true; }
  case EK_I32: { auto v = foldHost(static_cast<const std::int32_t *>(buffer), n, access, parallel); std::memcpy(out8, &v, 4); return true; }
  case EK_I64: { auto v = foldHost(static_cast<const std::int64_t *>(buffer), n, access, parallel); std::memcpy(out8, &v, 8); return true; }
  default: return false;
  }
}

bool reductionCombine(void *dst, const void *value, int kind, int access) {
  if (!validAccess(access))
    return false;
  auto go = [&](auto tag) {
    using T = decltype(tag);
    T a, v;
    std::memcpy(&a, dst, sizeof(T));
    std::memcpy(&v, value, sizeof(T));
    a = combine(access, a, v);
    std::memcpy(dst, &a, sizeof(T));
    return true;
  };
  switch (kind) {
  case EK_F32: return go(float{});
  case EK_F64: return go(double{});
  case EK_I32: return go(std::int32_t{});
  case EK_I64: return go(std::int64_t{});
  default: return false;
  }
}

//===----------------------------------------------------------------------===//
// CUDA: two small PTX kernels per element type and operation
//===----------------------------------------------------------------------===//

#ifdef OPS_ENABLE_CUDA
namespace {

constexpr int kBlock = 256;
constexpr int kMaxBlocks = 2048;

struct PtxType {
  const char *name; // PTX type
  int shift;        // log2 of the size
};

PtxType ptxType(int kind) {
  switch (kind) {
  case EK_F32: return {"f32", 2};
  case EK_F64: return {"f64", 3};
  case EK_I32: return {"s32", 2};
  default: return {"s64", 3};
  }
}

std::string replaceAll(std::string s, const std::string &from, const std::string &to) {
  for (std::size_t at = 0; (at = s.find(from, at)) != std::string::npos; at += to.size())
    s.replace(at, from.size(), to);
  return s;
}

// fill(buf, n, value): grid-stride store.  red(buf, n, out, identity): each thread accumulates a
// strided slice, the block folds its threads through shared memory, and thread 0 stores the
// block's result at out[blockIdx.x].  A second launch of red over the block results (one block)
// gives the answer.
const char *kPtxTemplate = R"(
.version 7.0
.target sm_70
.address_size 64

.visible .entry fill(.param .u64 p_buf, .param .u64 p_n, .param .@T@ p_val)
{
  .reg .pred %p<2>;
  .reg .b32 %r<8>;
  .reg .b64 %rd<10>;
  .reg .@T@ %v<2>;
  ld.param.u64 %rd1, [p_buf];
  ld.param.u64 %rd2, [p_n];
  ld.param.@T@ %v1, [p_val];
  mov.u32 %r1, %ctaid.x;
  mov.u32 %r2, %ntid.x;
  mov.u32 %r3, %tid.x;
  mov.u32 %r4, %nctaid.x;
  mad.lo.s32 %r5, %r1, %r2, %r3;
  mul.lo.s32 %r6, %r2, %r4;
  cvt.u64.u32 %rd3, %r5;
  cvt.u64.u32 %rd4, %r6;
$L_loop:
  setp.ge.u64 %p1, %rd3, %rd2;
  @%p1 bra $L_done;
  shl.b64 %rd5, %rd3, @S@;
  add.s64 %rd6, %rd1, %rd5;
  st.global.@T@ [%rd6], %v1;
  add.s64 %rd3, %rd3, %rd4;
  bra.uni $L_loop;
$L_done:
  ret;
}

.visible .entry red(.param .u64 p_buf, .param .u64 p_n, .param .u64 p_out, .param .@T@ p_id)
{
  .reg .pred %p<4>;
  .reg .b32 %r<12>;
  .reg .b64 %rd<14>;
  .reg .@T@ %v<6>;
  .shared .align 8 .b8 smem[2048];
  ld.param.u64 %rd1, [p_buf];
  ld.param.u64 %rd2, [p_n];
  ld.param.u64 %rd3, [p_out];
  ld.param.@T@ %v1, [p_id];
  mov.u32 %r1, %ctaid.x;
  mov.u32 %r2, %ntid.x;
  mov.u32 %r3, %tid.x;
  mov.u32 %r4, %nctaid.x;
  mad.lo.s32 %r5, %r1, %r2, %r3;
  mul.lo.s32 %r6, %r2, %r4;
  cvt.u64.u32 %rd4, %r5;
  cvt.u64.u32 %rd5, %r6;
$L_loop:
  setp.ge.u64 %p1, %rd4, %rd2;
  @%p1 bra $L_loopdone;
  shl.b64 %rd6, %rd4, @S@;
  add.s64 %rd7, %rd1, %rd6;
  ld.global.@T@ %v2, [%rd7];
  @OP@.@T@ %v1, %v1, %v2;
  add.s64 %rd4, %rd4, %rd5;
  bra.uni $L_loop;
$L_loopdone:
  mov.u64 %rd8, smem;
  cvt.u64.u32 %rd9, %r3;
  shl.b64 %rd10, %rd9, @S@;
  add.s64 %rd11, %rd8, %rd10;
  st.shared.@T@ [%rd11], %v1;
  bar.sync 0;
  mov.u32 %r7, 128;
$L_tree:
  setp.ge.u32 %p2, %r3, %r7;
  @%p2 bra $L_skip;
  add.u32 %r8, %r3, %r7;
  cvt.u64.u32 %rd12, %r8;
  shl.b64 %rd12, %rd12, @S@;
  add.s64 %rd13, %rd8, %rd12;
  ld.shared.@T@ %v3, [%rd13];
  @OP@.@T@ %v1, %v1, %v3;
  st.shared.@T@ [%rd11], %v1;
$L_skip:
  bar.sync 0;
  shr.u32 %r7, %r7, 1;
  setp.gt.u32 %p3, %r7, 0;
  @%p3 bra $L_tree;
  setp.ne.u32 %p2, %r3, 0;
  @%p2 bra $L_end;
  ld.shared.@T@ %v4, [%rd8];
  cvt.u64.u32 %rd9, %r1;
  shl.b64 %rd9, %rd9, @S@;
  add.s64 %rd9, %rd3, %rd9;
  st.global.@T@ [%rd9], %v4;
$L_end:
  ret;
}
)";

struct CudaKernels {
  CUmodule module = nullptr;
  CUfunction fill = nullptr, red = nullptr;
};

CudaKernels *cudaKernels(int kind, int access) {
  static std::mutex mu;
  static std::map<std::pair<int, int>, CudaKernels> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto [it, fresh] = cache.try_emplace({kind, access});
  CudaKernels &k = it->second;
  if (!fresh)
    return k.fill ? &k : nullptr;
  PtxType t = ptxType(kind);
  std::string ptx = replaceAll(kPtxTemplate, "@T@", t.name);
  ptx = replaceAll(ptx, "@S@", std::to_string(t.shift));
  ptx = replaceAll(ptx, "@OP@", access == kInc ? "add" : access == kMin ? "min" : "max");
  if (cuModuleLoadData(&k.module, ptx.c_str()) != CUDA_SUCCESS ||
      cuModuleGetFunction(&k.fill, k.module, "fill") != CUDA_SUCCESS ||
      cuModuleGetFunction(&k.red, k.module, "red") != CUDA_SUCCESS) {
    k.fill = k.red = nullptr;
    return nullptr;
  }
  return &k;
}

unsigned blocksFor(std::size_t n) {
  return static_cast<unsigned>(std::max<std::size_t>(1, std::min<std::size_t>((n + kBlock - 1) / kBlock, kMaxBlocks)));
}

} // namespace

std::size_t reductionScratchBytes() { return (kMaxBlocks + 1) * 8; }

bool reductionFillCuda(CUdeviceptr buffer, std::size_t n, int kind, int access, CUstream stream) {
  if (!validAccess(access))
    return false;
  CudaKernels *k = cudaKernels(kind, access);
  if (!k)
    return false;
  alignas(8) char id[8];
  reductionIdentity(kind, access, id);
  unsigned long long count = n;
  // the value parameter is passed by its bytes, whatever its type
  void *params[] = {&buffer, &count, id};
  return cuLaunchKernel(k->fill, blocksFor(n), 1, 1, kBlock, 1, 1, 0, stream, params, nullptr) == CUDA_SUCCESS;
}

bool reductionFoldCuda(CUdeviceptr buffer, std::size_t n, int kind, int access, CUstream stream,
                       CUdeviceptr scratch, void *out8) {
  if (!validAccess(access))
    return false;
  CudaKernels *k = cudaKernels(kind, access);
  if (!k)
    return false;
  alignas(8) char id[8];
  reductionIdentity(kind, access, id);
  CUdeviceptr partial = scratch, result = scratch + kMaxBlocks * 8;
  unsigned long long count = n;
  void *first[] = {&buffer, &count, &partial, id};
  const unsigned blocks = blocksFor(n);
  if (cuLaunchKernel(k->red, blocks, 1, 1, kBlock, 1, 1, 0, stream, first, nullptr) != CUDA_SUCCESS)
    return false;
  unsigned long long nb = blocks;
  void *second[] = {&partial, &nb, &result, id};
  if (cuLaunchKernel(k->red, 1, 1, 1, kBlock, 1, 1, 0, stream, second, nullptr) != CUDA_SUCCESS)
    return false;
  std::memset(out8, 0, 8);
  return cuMemcpyDtoH(out8, result, ptxType(kind).shift == 3 ? 8 : 4) == CUDA_SUCCESS;
}
#endif // OPS_ENABLE_CUDA

} // namespace ops_mlir
