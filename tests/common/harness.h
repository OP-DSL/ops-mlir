// Minimal e2e harness shared by the fusion/precision tests.
#ifndef OPS_MLIR_TEST_HARNESS_H
#define OPS_MLIR_TEST_HARNESS_H

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace harness {

inline int failures = 0;
inline int passes = 0;

inline void report(bool ok, const std::string &name, const std::string &detail = "") {
  if (ok) {
    ++passes;
    std::printf("[PASS] %s\n", name.c_str());
  } else {
    ++failures;
    std::printf("[FAIL] %s: %s\n", name.c_str(), detail.c_str());
  }
  std::fflush(stdout);
}

// Comparison tolerance: C * eps * steps, applied relative to (1 + |ref|).
template <typename T>
double tolerance(int steps = 1, double c = 16.0) {
  return c * std::numeric_limits<T>::epsilon() * (steps < 1 ? 1 : steps);
}

// A 2D field with a one-cell halo on every side, indexed by OPS global
// coordinates i in [-1, nx], j in [-1, ny]; x (i) is the unit-stride axis.
template <typename T> struct Field {
  int nx, ny, sx; // sx = allocated x extent (row stride)
  const T *p;
  double at(int i, int j) const { return p[(j + 1) * sx + (i + 1)]; }
};

// Compare a device/host field against a double-precision reference over
// [i0,i1) x [j0,j1). Returns true on success; fills `detail` otherwise.
template <typename T>
bool compare(const Field<T> &f, const std::function<double(int, int)> &ref,
             int i0, int i1, int j0, int j1, double tol, std::string &detail) {
  double worst = 0.0;
  int wi = 0, wj = 0;
  double wgot = 0.0, wref = 0.0;
  for (int j = j0; j < j1; ++j)
    for (int i = i0; i < i1; ++i) {
      double r = ref(i, j), g = f.at(i, j);
      double e = std::fabs(g - r) / (1.0 + std::fabs(r));
      if (!(e <= worst)) { // also catches NaN
        worst = e; wi = i; wj = j; wgot = g; wref = r;
      }
    }
  if (worst <= tol)
    return true;
  char buf[256];
  std::snprintf(buf, sizeof buf,
                "max rel err %.3e > tol %.3e at (%d,%d): got %.9g want %.9g",
                worst, tol, wi, wj, wgot, wref);
  detail = buf;
  return false;
}

// Bitwise comparison of two fields (fused vs. unfused on CPU backends).
template <typename T>
bool identical(const Field<T> &a, const Field<T> &b, int i0, int i1, int j0,
               int j1) {
  for (int j = j0; j < j1; ++j)
    for (int i = i0; i < i1; ++i) {
      T x = a.at(i, j), y = b.at(i, j);
      if (std::memcmp(&x, &y, sizeof(T)) != 0)
        return false;
    }
  return true;
}

} // namespace harness

#endif
