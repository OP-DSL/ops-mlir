// Entry point for CloverLeaf 2D under ops-mlir. The original main() (clover_leaf.cpp,
// compiled with -Dmain=clover_original_main) does the real work; this registers the
// kernel headers the JIT translates kernels from, then calls it.
#include "ops/OPSWrapper.h"

#include <sstream>
#include <string>

int clover_original_main(int argc, char **argv);

int main(int argc, char **argv) {
  // CLOVER_KERNEL_HEADERS: ':'-separated absolute paths, set by CMake.
  std::stringstream headers(CLOVER_KERNEL_HEADERS);
  std::string path;
  while (std::getline(headers, path, ':'))
    if (!path.empty())
      set_kernel_source_file(path);
  // The kernel headers rely on what the application's own sources include before them.
  set_kernel_preamble("#include \"data.h\"\n#include \"definitions.h\"\n");
  return clover_original_main(argc, argv);
}
