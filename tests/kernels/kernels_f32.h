// Single-precision instantiation of the shared e2e test kernels.
typedef float real_t;
#define RLIT(x) x##f
#include "kernels_impl.h"
