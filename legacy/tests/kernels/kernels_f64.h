// Double-precision instantiation of the shared e2e test kernels.
// KernelIRBuilder parses this file with fixed clang args (no -D), so the
// precision is selected by which thin header the app registers.
typedef double real_t;
#define RLIT(x) x
#include "kernels_impl.h"
