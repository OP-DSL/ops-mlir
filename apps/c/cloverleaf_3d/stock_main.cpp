// The stock build of CloverLeaf 3D: clover_leaf.cpp's main() is renamed for the ops-mlir
// build (a per-source property, so it applies to this target as well); forward to it.
int clover_original_main(int argc, const char **argv);
int main(int argc, const char **argv) { return clover_original_main(argc, argv); }
